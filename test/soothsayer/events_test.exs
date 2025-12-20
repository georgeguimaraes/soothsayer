defmodule Soothsayer.EventsTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Events
  alias Explorer.DataFrame
  alias Explorer.Series

  describe "n_features/1" do
    test "returns 0 for empty events config" do
      assert Events.n_features(%{}) == 0
    end

    test "returns 1 for single event with no window" do
      config = %{"sale" => %{lower_window: 0, upper_window: 0}}
      assert Events.n_features(config) == 1
    end

    test "counts window positions correctly" do
      # lower_window: -2, upper_window: 1 = positions [-2, -1, 0, +1] = 4 features
      config = %{"black_friday" => %{lower_window: -2, upper_window: 1}}
      assert Events.n_features(config) == 4
    end

    test "sums features across multiple events" do
      config = %{
        "black_friday" => %{lower_window: -2, upper_window: 1},  # 4 features
        "christmas" => %{lower_window: -1, upper_window: 0}       # 2 features
      }
      assert Events.n_features(config) == 6
    end
  end

  describe "feature_names/1" do
    test "returns empty list for empty config" do
      assert Events.feature_names(%{}) == []
    end

    test "returns single name for event with no window" do
      config = %{"sale" => %{lower_window: 0, upper_window: 0}}
      assert Events.feature_names(config) == ["sale_0"]
    end

    test "returns names for all window positions" do
      config = %{"black_friday" => %{lower_window: -2, upper_window: 1}}
      names = Events.feature_names(config)

      assert length(names) == 4
      assert "black_friday_-2" in names
      assert "black_friday_-1" in names
      assert "black_friday_0" in names
      assert "black_friday_+1" in names
    end

    test "returns names for multiple events sorted by event name" do
      config = %{
        "christmas" => %{lower_window: 0, upper_window: 0},
        "black_friday" => %{lower_window: -1, upper_window: 0}
      }
      names = Events.feature_names(config)

      # Should be sorted by event name, then by window position
      assert names == ["black_friday_-1", "black_friday_0", "christmas_0"]
    end
  end

  describe "build_features/3" do
    test "returns tensor with correct shape" do
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])

      events_df = DataFrame.new(%{
        "event" => ["sale"],
        "ds" => [~D[2023-01-02]]
      })

      config = %{"sale" => %{lower_window: 0, upper_window: 0}}

      tensor = Events.build_features(dates, events_df, config)

      assert Nx.shape(tensor) == {3, 1}
    end

    test "returns nil for empty events config" do
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02]])
      events_df = DataFrame.new(%{"event" => [], "ds" => []})
      config = %{}

      # When no events configured, return nil (caller should skip events input)
      assert Events.build_features(dates, events_df, config) == nil
    end

    test "sets 1.0 for exact event date with window 0" do
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])

      events_df = DataFrame.new(%{
        "event" => ["sale"],
        "ds" => [~D[2023-01-02]]
      })

      config = %{"sale" => %{lower_window: 0, upper_window: 0}}

      tensor = Events.build_features(dates, events_df, config)

      # Row 0 (Jan 1): no event
      # Row 1 (Jan 2): sale event
      # Row 2 (Jan 3): no event
      expected = Nx.tensor([[0.0], [1.0], [0.0]])
      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles multiple events" do
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])

      events_df = DataFrame.new(%{
        "event" => ["event_a", "event_b"],
        "ds" => [~D[2023-01-01], ~D[2023-01-03]]
      })

      config = %{
        "event_a" => %{lower_window: 0, upper_window: 0},
        "event_b" => %{lower_window: 0, upper_window: 0}
      }

      tensor = Events.build_features(dates, events_df, config)

      assert Nx.shape(tensor) == {3, 2}

      # Columns sorted by event name: event_a, event_b
      # Row 0 (Jan 1): event_a=1, event_b=0
      # Row 1 (Jan 2): event_a=0, event_b=0
      # Row 2 (Jan 3): event_a=0, event_b=1
      expected = Nx.tensor([
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0]
      ])
      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles recurring events (same event on multiple dates)" do
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])

      events_df = DataFrame.new(%{
        "event" => ["sale", "sale"],
        "ds" => [~D[2023-01-01], ~D[2023-01-03]]
      })

      config = %{"sale" => %{lower_window: 0, upper_window: 0}}

      tensor = Events.build_features(dates, events_df, config)

      expected = Nx.tensor([[1.0], [0.0], [1.0]])
      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles lower_window (days before event)" do
      # Event on Jan 3, with lower_window: -2
      # Should create feature columns for: -2 (Jan 1), -1 (Jan 2), 0 (Jan 3)
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04]])

      events_df = DataFrame.new(%{
        "event" => ["sale"],
        "ds" => [~D[2023-01-03]]
      })

      config = %{"sale" => %{lower_window: -2, upper_window: 0}}

      tensor = Events.build_features(dates, events_df, config)

      # Shape: {4 dates, 3 features: sale_-2, sale_-1, sale_0}
      assert Nx.shape(tensor) == {4, 3}

      # Column 0 (sale_-2): 1.0 on Jan 1 (2 days before Jan 3)
      # Column 1 (sale_-1): 1.0 on Jan 2 (1 day before Jan 3)
      # Column 2 (sale_0):  1.0 on Jan 3 (event day)
      expected = Nx.tensor([
        [1.0, 0.0, 0.0],  # Jan 1: sale_-2
        [0.0, 1.0, 0.0],  # Jan 2: sale_-1
        [0.0, 0.0, 1.0],  # Jan 3: sale_0
        [0.0, 0.0, 0.0]   # Jan 4: nothing
      ])
      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles upper_window (days after event)" do
      # Event on Jan 1, with upper_window: 2
      # Should create feature columns for: 0 (Jan 1), +1 (Jan 2), +2 (Jan 3)
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04]])

      events_df = DataFrame.new(%{
        "event" => ["sale"],
        "ds" => [~D[2023-01-01]]
      })

      config = %{"sale" => %{lower_window: 0, upper_window: 2}}

      tensor = Events.build_features(dates, events_df, config)

      # Shape: {4 dates, 3 features: sale_0, sale_+1, sale_+2}
      assert Nx.shape(tensor) == {4, 3}

      # Column 0 (sale_0):  1.0 on Jan 1 (event day)
      # Column 1 (sale_+1): 1.0 on Jan 2 (1 day after Jan 1)
      # Column 2 (sale_+2): 1.0 on Jan 3 (2 days after Jan 1)
      expected = Nx.tensor([
        [1.0, 0.0, 0.0],  # Jan 1: sale_0
        [0.0, 1.0, 0.0],  # Jan 2: sale_+1
        [0.0, 0.0, 1.0],  # Jan 3: sale_+2
        [0.0, 0.0, 0.0]   # Jan 4: nothing
      ])
      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles both lower and upper windows" do
      # Event on Jan 3, with lower_window: -1, upper_window: 1
      # Should create features for: -1 (Jan 2), 0 (Jan 3), +1 (Jan 4)
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04], ~D[2023-01-05]])

      events_df = DataFrame.new(%{
        "event" => ["sale"],
        "ds" => [~D[2023-01-03]]
      })

      config = %{"sale" => %{lower_window: -1, upper_window: 1}}

      tensor = Events.build_features(dates, events_df, config)

      # Shape: {5 dates, 3 features: sale_-1, sale_0, sale_+1}
      assert Nx.shape(tensor) == {5, 3}

      expected = Nx.tensor([
        [0.0, 0.0, 0.0],  # Jan 1: nothing
        [1.0, 0.0, 0.0],  # Jan 2: sale_-1 (1 day before event)
        [0.0, 1.0, 0.0],  # Jan 3: sale_0 (event day)
        [0.0, 0.0, 1.0],  # Jan 4: sale_+1 (1 day after event)
        [0.0, 0.0, 0.0]   # Jan 5: nothing
      ])
      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles overlapping windows from different events" do
      # event_a on Jan 2, event_b on Jan 3, both with window -1 to 0
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04]])

      events_df = DataFrame.new(%{
        "event" => ["event_a", "event_b"],
        "ds" => [~D[2023-01-02], ~D[2023-01-03]]
      })

      config = %{
        "event_a" => %{lower_window: -1, upper_window: 0},
        "event_b" => %{lower_window: -1, upper_window: 0}
      }

      tensor = Events.build_features(dates, events_df, config)

      # Shape: {4 dates, 4 features: event_a_-1, event_a_0, event_b_-1, event_b_0}
      assert Nx.shape(tensor) == {4, 4}

      # Columns sorted by event name, then window position
      expected = Nx.tensor([
        [1.0, 0.0, 0.0, 0.0],  # Jan 1: event_a_-1
        [0.0, 1.0, 1.0, 0.0],  # Jan 2: event_a_0, event_b_-1
        [0.0, 0.0, 0.0, 1.0],  # Jan 3: event_b_0
        [0.0, 0.0, 0.0, 0.0]   # Jan 4: nothing
      ])
      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end
  end
end
