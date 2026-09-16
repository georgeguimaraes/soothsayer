defmodule Soothsayer.EventsTest do
  use ExUnit.Case, async: true

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Events

  describe "build_network_input/1" do
    test "returns nil when no events configured" do
      config = %{}
      assert Events.build_network_input(config) == nil
    end

    test "returns nil when events key is empty map" do
      config = %{events: %{}}
      assert Events.build_network_input(config) == nil
    end

    test "returns Axon input with correct shape when events configured" do
      config = %{
        events: %{
          "sale" => %{steps_before: 0, steps_after: 0},
          "holiday" => %{steps_before: 1, steps_after: 1}
        }
      }

      input = Events.build_network_input(config)

      # sale: 1 feature, holiday: 3 features = 4 total
      assert Axon.get_inputs(input)["events"] == {nil, 1, 4}
    end
  end

  describe "build_component/2" do
    test "returns constant 0 when no events configured" do
      config = %{}
      input = Axon.input("events", shape: {nil, 1, 1})

      component = Events.build_component(input, config)

      {init_fn, predict_fn} = Axon.build(component)
      params = init_fn.(%{"events" => Nx.tensor([[[1.0]]])}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"events" => Nx.tensor([[[1.0]]])})

      assert Nx.to_number(output) == 0.0
    end

    test "returns dense layer when events configured" do
      config = %{events: %{"sale" => %{steps_before: 0, steps_after: 0}}}
      input = Axon.input("events", shape: {nil, 1, 1})

      component = Events.build_component(input, config)

      {init_fn, _predict_fn} = Axon.build(component)
      params = init_fn.(%{"events" => Nx.tensor([[[1.0]]])}, Axon.ModelState.empty())

      assert Map.has_key?(params.data, "events_dense")
    end
  end

  describe "n_features/1" do
    test "returns 0 for empty events config" do
      assert Events.n_features(%{}) == 0
    end

    test "returns 1 for single event with no window" do
      config = %{"sale" => %{steps_before: 0, steps_after: 0}}
      assert Events.n_features(config) == 1
    end

    test "counts window positions correctly" do
      # steps_before: 2, steps_after: 1 = positions [-2, -1, 0, +1] = 4 features
      config = %{"black_friday" => %{steps_before: 2, steps_after: 1}}
      assert Events.n_features(config) == 4
    end

    test "sums features across multiple events" do
      config = %{
        # 4 features
        "black_friday" => %{steps_before: 2, steps_after: 1},
        # 2 features
        "christmas" => %{steps_before: 1, steps_after: 0}
      }

      assert Events.n_features(config) == 6
    end
  end

  describe "feature_names/1" do
    test "returns empty list for empty config" do
      assert Events.feature_names(%{}) == []
    end

    test "returns single name for event with no window" do
      config = %{"sale" => %{steps_before: 0, steps_after: 0}}
      assert Events.feature_names(config) == ["sale_0"]
    end

    test "returns names for all window positions" do
      config = %{"black_friday" => %{steps_before: 2, steps_after: 1}}
      names = Events.feature_names(config)

      assert length(names) == 4
      assert "black_friday_-2" in names
      assert "black_friday_-1" in names
      assert "black_friday_0" in names
      assert "black_friday_+1" in names
    end

    test "returns names for multiple events sorted by event name" do
      config = %{
        "christmas" => %{steps_before: 0, steps_after: 0},
        "black_friday" => %{steps_before: 1, steps_after: 0}
      }

      names = Events.feature_names(config)

      # Should be sorted by event name, then by window position
      assert names == ["black_friday_-1", "black_friday_0", "christmas_0"]
    end
  end

  describe "build_features/3" do
    test "returns tensor with correct shape" do
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])

      events_df =
        DataFrame.new(%{
          "event" => ["sale"],
          "ds" => [~D[2023-01-02]]
        })

      config = %{"sale" => %{steps_before: 0, steps_after: 0}}

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

      events_df =
        DataFrame.new(%{
          "event" => ["sale"],
          "ds" => [~D[2023-01-02]]
        })

      config = %{"sale" => %{steps_before: 0, steps_after: 0}}

      tensor = Events.build_features(dates, events_df, config)

      # Row 0 (Jan 1): no event
      # Row 1 (Jan 2): sale event
      # Row 2 (Jan 3): no event
      expected = Nx.tensor([[0.0], [1.0], [0.0]])
      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles multiple events" do
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])

      events_df =
        DataFrame.new(%{
          "event" => ["event_a", "event_b"],
          "ds" => [~D[2023-01-01], ~D[2023-01-03]]
        })

      config = %{
        "event_a" => %{steps_before: 0, steps_after: 0},
        "event_b" => %{steps_before: 0, steps_after: 0}
      }

      tensor = Events.build_features(dates, events_df, config)

      assert Nx.shape(tensor) == {3, 2}

      # Columns sorted by event name: event_a, event_b
      # Row 0 (Jan 1): event_a=1, event_b=0
      # Row 1 (Jan 2): event_a=0, event_b=0
      # Row 2 (Jan 3): event_a=0, event_b=1
      expected =
        Nx.tensor([
          [1.0, 0.0],
          [0.0, 0.0],
          [0.0, 1.0]
        ])

      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles recurring events (same event on multiple dates)" do
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])

      events_df =
        DataFrame.new(%{
          "event" => ["sale", "sale"],
          "ds" => [~D[2023-01-01], ~D[2023-01-03]]
        })

      config = %{"sale" => %{steps_before: 0, steps_after: 0}}

      tensor = Events.build_features(dates, events_df, config)

      expected = Nx.tensor([[1.0], [0.0], [1.0]])
      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles steps_before" do
      # Event on Jan 3, with steps_before: 2
      # Should create feature columns for: -2 (Jan 1), -1 (Jan 2), 0 (Jan 3)
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04]])

      events_df =
        DataFrame.new(%{
          "event" => ["sale"],
          "ds" => [~D[2023-01-03]]
        })

      config = %{"sale" => %{steps_before: 2, steps_after: 0}}

      tensor = Events.build_features(dates, events_df, config)

      # Shape: {4 dates, 3 features: sale_-2, sale_-1, sale_0}
      assert Nx.shape(tensor) == {4, 3}

      # Column 0 (sale_-2): 1.0 on Jan 1 (2 days before Jan 3)
      # Column 1 (sale_-1): 1.0 on Jan 2 (1 day before Jan 3)
      # Column 2 (sale_0):  1.0 on Jan 3 (event day)
      expected =
        Nx.tensor([
          # Jan 1: sale_-2
          [1.0, 0.0, 0.0],
          # Jan 2: sale_-1
          [0.0, 1.0, 0.0],
          # Jan 3: sale_0
          [0.0, 0.0, 1.0],
          # Jan 4: nothing
          [0.0, 0.0, 0.0]
        ])

      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles steps_after" do
      # Event on Jan 1, with steps_after: 2
      # Should create feature columns for: 0 (Jan 1), +1 (Jan 2), +2 (Jan 3)
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04]])

      events_df =
        DataFrame.new(%{
          "event" => ["sale"],
          "ds" => [~D[2023-01-01]]
        })

      config = %{"sale" => %{steps_before: 0, steps_after: 2}}

      tensor = Events.build_features(dates, events_df, config)

      # Shape: {4 dates, 3 features: sale_0, sale_+1, sale_+2}
      assert Nx.shape(tensor) == {4, 3}

      # Column 0 (sale_0):  1.0 on Jan 1 (event day)
      # Column 1 (sale_+1): 1.0 on Jan 2 (1 day after Jan 1)
      # Column 2 (sale_+2): 1.0 on Jan 3 (2 days after Jan 1)
      expected =
        Nx.tensor([
          # Jan 1: sale_0
          [1.0, 0.0, 0.0],
          # Jan 2: sale_+1
          [0.0, 1.0, 0.0],
          # Jan 3: sale_+2
          [0.0, 0.0, 1.0],
          # Jan 4: nothing
          [0.0, 0.0, 0.0]
        ])

      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles both lower and upper windows" do
      # Event on Jan 3, with steps_before: 1, steps_after: 1
      # Should create features for: -1 (Jan 2), 0 (Jan 3), +1 (Jan 4)
      dates =
        Series.from_list([
          ~D[2023-01-01],
          ~D[2023-01-02],
          ~D[2023-01-03],
          ~D[2023-01-04],
          ~D[2023-01-05]
        ])

      events_df =
        DataFrame.new(%{
          "event" => ["sale"],
          "ds" => [~D[2023-01-03]]
        })

      config = %{"sale" => %{steps_before: 1, steps_after: 1}}

      tensor = Events.build_features(dates, events_df, config)

      # Shape: {5 dates, 3 features: sale_-1, sale_0, sale_+1}
      assert Nx.shape(tensor) == {5, 3}

      expected =
        Nx.tensor([
          # Jan 1: nothing
          [0.0, 0.0, 0.0],
          # Jan 2: sale_-1 (1 day before event)
          [1.0, 0.0, 0.0],
          # Jan 3: sale_0 (event day)
          [0.0, 1.0, 0.0],
          # Jan 4: sale_+1 (1 day after event)
          [0.0, 0.0, 1.0],
          # Jan 5: nothing
          [0.0, 0.0, 0.0]
        ])

      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "handles overlapping windows from different events" do
      # event_a on Jan 2, event_b on Jan 3, both with window -1 to 0
      dates = Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04]])

      events_df =
        DataFrame.new(%{
          "event" => ["event_a", "event_b"],
          "ds" => [~D[2023-01-02], ~D[2023-01-03]]
        })

      config = %{
        "event_a" => %{steps_before: 1, steps_after: 0},
        "event_b" => %{steps_before: 1, steps_after: 0}
      }

      tensor = Events.build_features(dates, events_df, config)

      # Shape: {4 dates, 4 features: event_a_-1, event_a_0, event_b_-1, event_b_0}
      assert Nx.shape(tensor) == {4, 4}

      # Columns sorted by event name, then window position
      expected =
        Nx.tensor([
          # Jan 1: event_a_-1
          [1.0, 0.0, 0.0, 0.0],
          # Jan 2: event_a_0, event_b_-1
          [0.0, 1.0, 1.0, 0.0],
          # Jan 3: event_b_0
          [0.0, 0.0, 0.0, 1.0],
          # Jan 4: nothing
          [0.0, 0.0, 0.0, 0.0]
        ])

      assert Nx.equal(tensor, expected) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "event_dates/3" do
    test "repeats yearly events on their month and day over the years asked for" do
      config = %{events: %{"launch" => %{steps_before: 0, steps_after: 0, recurring: :yearly}}}
      events_df = DataFrame.new(%{"event" => ["launch"], "ds" => [~D[2022-03-01]]})
      timestamps = [~N[2022-01-01 00:00:00], ~N[2023-12-31 00:00:00]]

      assert Events.event_dates(events_df, config, timestamps) ==
               %{"launch" => [~N[2022-03-01 00:00:00], ~N[2023-03-01 00:00:00]]}
    end

    test "a leap day recurs only in leap years and keeps its time of day" do
      config = %{events: %{"leap" => %{steps_before: 0, steps_after: 0, recurring: :yearly}}}
      events_df = DataFrame.new(%{"event" => ["leap"], "ds" => [~N[2020-02-29 09:00:00]]})
      timestamps = Enum.map(2020..2024, &NaiveDateTime.new!(&1, 6, 1, 0, 0, 0))

      assert Events.event_dates(events_df, config, timestamps) ==
               %{"leap" => [~N[2020-02-29 09:00:00], ~N[2024-02-29 09:00:00]]}
    end

    test "unions what the model remembers with the frame, and ignores unconfigured events" do
      config = %{
        events: %{"sale" => %{steps_before: 0, steps_after: 0}},
        training_data: %{
          event_dates: %{"sale" => [~N[2022-05-01 00:00:00]], "old" => [~N[2022-01-01 00:00:00]]}
        }
      }

      events_df =
        DataFrame.new(%{"event" => ["sale", "other"], "ds" => [~D[2023-05-01], ~D[2023-06-01]]})

      timestamps = [~N[2023-01-01 00:00:00]]

      assert Events.event_dates(events_df, config, timestamps) ==
               %{"sale" => [~N[2022-05-01 00:00:00], ~N[2023-05-01 00:00:00]]}

      assert Events.event_dates(nil, config, timestamps) == %{"sale" => [~N[2022-05-01 00:00:00]]}
    end

    test "adds the country holidays named on the config as midnight timestamps" do
      config = %{
        events: %{"Christmas Day" => %{steps_before: 0, steps_after: 0}},
        holidays: %{
          countries: [:us],
          steps_before: 0,
          steps_after: 0,
          regions: [],
          include_informal: false,
          names: ["Christmas Day"]
        }
      }

      timestamps = [~N[2022-06-01 12:00:00], ~N[2023-06-01 12:00:00]]

      assert Events.event_dates(nil, config, timestamps) ==
               %{"Christmas Day" => [~N[2022-12-25 00:00:00], ~N[2023-12-25 00:00:00]]}
    end
  end

  describe "build_features/4 with a frequency" do
    test "windows are steps of the frequency and event dates mean midnight" do
      hours = Enum.map(0..47, &NaiveDateTime.add(~N[2023-01-04 00:00:00], &1, :hour))
      events_df = DataFrame.new(%{"event" => ["sale"], "ds" => [~D[2023-01-05]]})
      config = %{"sale" => %{steps_before: 1, steps_after: 1}}

      tensor = Events.build_features(Series.from_list(hours), events_df, config, {1, :hour})

      assert Nx.shape(tensor) == {48, 3}

      # Column offsets -1, 0, +1 fire at 23:00 on the 4th, 00:00 and 01:00 on the 5th.
      active_rows =
        tensor
        |> Nx.sum(axes: [1])
        |> Nx.to_flat_list()
        |> Enum.with_index()
        |> Enum.filter(fn {value, _index} -> value == 1.0 end)
        |> Enum.map(fn {_value, index} -> Enum.at(hours, index) end)

      assert active_rows == [
               ~N[2023-01-04 23:00:00],
               ~N[2023-01-05 00:00:00],
               ~N[2023-01-05 01:00:00]
             ]

      assert Nx.to_flat_list(tensor[23]) == [1.0, 0.0, 0.0]
      assert Nx.to_flat_list(tensor[24]) == [0.0, 1.0, 0.0]
      assert Nx.to_flat_list(tensor[25]) == [0.0, 0.0, 1.0]
    end
  end

  describe "get_effects/1" do
    test "raises when no events configured" do
      model = %Soothsayer.Model{
        config: %{},
        params: %Axon.ModelState{data: %{}},
        network: nil
      }

      assert_raise ArgumentError, "No events configured on this model", fn ->
        Events.get_effects(model)
      end
    end

    test "raises when model not fitted" do
      model = %Soothsayer.Model{
        config: %{events: %{"sale" => %{steps_before: 0, steps_after: 0}}},
        params: nil,
        network: nil
      }

      assert_raise ArgumentError, "Model has not been fitted yet", fn ->
        Events.get_effects(model)
      end
    end

    test "returns map of feature names to coefficients" do
      # Create mock params with events_dense layer
      kernel = Nx.tensor([[1.5], [2.5]])
      bias = Nx.tensor([0.0])

      model = %Soothsayer.Model{
        config: %{events: %{"sale" => %{steps_before: 1, steps_after: 0}}},
        params: %Axon.ModelState{data: %{"events_dense" => %{"kernel" => kernel, "bias" => bias}}},
        network: nil
      }

      effects = Events.get_effects(model)

      assert is_map(effects)
      assert Map.has_key?(effects, "sale_-1")
      assert Map.has_key?(effects, "sale_0")
      assert_in_delta effects["sale_-1"], 1.5, 0.001
      assert_in_delta effects["sale_0"], 2.5, 0.001
    end
  end
end
