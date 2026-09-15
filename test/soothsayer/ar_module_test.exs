defmodule Soothsayer.ARModuleTest do
  use ExUnit.Case, async: true

  alias Soothsayer.AR

  describe "create_lagged_inputs/2" do
    test "creates sliding windows from y values" do
      y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      lags = 3

      {lagged, targets} = AR.create_lagged_inputs(y, lags)

      # With lags=3 and 5 values, we get 2 windows:
      # Window 1: [1, 2, 3] -> target: 4
      # Window 2: [2, 3, 4] -> target: 5
      assert Nx.shape(lagged) == {2, 3}
      assert Nx.shape(targets) == {2, 1}
      assert Nx.to_flat_list(lagged) == [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]
      assert Nx.to_flat_list(targets) == [4.0, 5.0]
    end
  end

  describe "known_values/1" do
    test "zips training dates with normalized values" do
      training_data = %{
        timestamps: [~D[2023-01-01], ~D[2023-01-02]],
        y_normalized: [0.5, -0.5]
      }

      assert AR.known_values(training_data) == %{~D[2023-01-01] => 0.5, ~D[2023-01-02] => -0.5}
    end
  end

  describe "build_input/4" do
    test "looks up the lags ending at each origin, oldest first" do
      known_values = %{
        ~D[2023-01-01] => 1.0,
        ~D[2023-01-02] => 2.0,
        ~D[2023-01-03] => 3.0,
        ~D[2023-01-04] => 4.0,
        ~D[2023-01-05] => 5.0
      }

      result =
        AR.build_input(
          known_values,
          [~D[2023-01-03], ~D[2023-01-04], ~D[2023-01-05]],
          2,
          {1, :day}
        )

      # origin 01-03 uses [01-02, 01-03], 01-04 uses [01-03, 01-04], 01-05 uses [01-04, 01-05]
      assert Nx.shape(result) == {3, 2}
      assert Nx.to_flat_list(result) == [2.0, 3.0, 3.0, 4.0, 4.0, 5.0]
    end

    test "returns zeros when any lagged day is unknown" do
      known_values = %{~D[2023-01-01] => 1.0, ~D[2023-01-02] => 2.0, ~D[2023-01-04] => 4.0}

      # 01-01 lacks 12-31, 01-04 lacks 01-03 (a gap), 01-02 has both
      result =
        AR.build_input(
          known_values,
          [~D[2023-01-01], ~D[2023-01-04], ~D[2023-01-02]],
          2,
          {1, :day}
        )

      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    end
  end

  describe "training_rows/3" do
    test "builds one row per origin and step, ordered step by step" do
      y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

      rows = AR.training_rows(y, 2, 2)

      # origins end at positions 1, 2, 3 (values 2, 3, 4); step 1 targets 3, 4, 5; step 2 targets 4, 5, 6
      assert Nx.to_list(rows.lagged) == [
               [1.0, 2.0],
               [2.0, 3.0],
               [3.0, 4.0],
               [1.0, 2.0],
               [2.0, 3.0],
               [3.0, 4.0]
             ]

      assert Nx.to_flat_list(rows.targets) == [3.0, 4.0, 5.0, 4.0, 5.0, 6.0]
      assert rows.target_indices == [2, 3, 4, 3, 4, 5]

      assert Nx.to_list(rows.step_mask) == [
               [1.0, 0.0],
               [1.0, 0.0],
               [1.0, 0.0],
               [0.0, 1.0],
               [0.0, 1.0],
               [0.0, 1.0]
             ]
    end

    test "one forecast step has no mask and matches create_lagged_inputs" do
      y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      rows = AR.training_rows(y, 3, 1)
      {lagged, targets} = AR.create_lagged_inputs(y, 3)

      assert rows.step_mask == nil
      assert rows.target_indices == [3, 4]
      assert Nx.to_list(rows.lagged) == Nx.to_list(lagged)
      assert Nx.to_list(rows.targets) == Nx.to_list(targets)
    end

    test "raises when the series is too short for the lags and steps" do
      assert_raise ArgumentError, ~r/need at least 5 rows, got 4/, fn ->
        AR.training_rows(Nx.tensor([1.0, 2.0, 3.0, 4.0]), 3, 2)
      end
    end
  end

  describe "origin_and_step/4" do
    test "observed dates are one step from the day before, later dates go in blocks" do
      last_observed_date = ~D[2023-01-31]

      daily = {1, :day}

      assert AR.origin_and_step(~D[2023-01-10], last_observed_date, 3, daily) ==
               {~D[2023-01-09], 1}

      assert AR.origin_and_step(~D[2023-01-31], last_observed_date, 3, daily) ==
               {~D[2023-01-30], 1}

      assert AR.origin_and_step(~D[2023-02-01], last_observed_date, 3, daily) ==
               {~D[2023-01-31], 1}

      assert AR.origin_and_step(~D[2023-02-03], last_observed_date, 3, daily) ==
               {~D[2023-01-31], 3}

      assert AR.origin_and_step(~D[2023-02-04], last_observed_date, 3, daily) ==
               {~D[2023-02-03], 1}

      assert AR.origin_and_step(~D[2023-02-07], last_observed_date, 3, daily) ==
               {~D[2023-02-06], 1}
    end

    test "with one step every date comes from the day before it" do
      assert AR.origin_and_step(~D[2023-02-09], ~D[2023-01-31], 1, {1, :day}) ==
               {~D[2023-02-08], 1}
    end

    test "steps follow the frequency and off-grid timestamps raise" do
      last_observed = ~N[2023-01-31 00:00:00]

      assert AR.origin_and_step(~N[2023-01-31 02:00:00], last_observed, 3, {1, :hour}) ==
               {~N[2023-01-31 00:00:00], 2}

      assert AR.origin_and_step(~N[2023-01-31 04:00:00], last_observed, 3, {1, :hour}) ==
               {~N[2023-01-31 03:00:00], 1}

      assert_raise ArgumentError, ~r/not a whole number of 1 hour steps/, fn ->
        AR.origin_and_step(~N[2023-01-31 01:30:00], last_observed, 3, {1, :hour})
      end
    end
  end

  describe "step_mask/2" do
    test "one-hot encodes 1-based steps and is nil for a single step" do
      assert AR.step_mask([1, 3, 2], 3) |> Nx.to_list() == [
               [1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0],
               [0.0, 1.0, 0.0]
             ]

      assert AR.step_mask([1, 1], 1) == nil
    end
  end

  describe "get_weights/1" do
    test "returns weights for linear AR model" do
      :rand.seed(:exsss, {42, 42, 42})

      n_points = 50
      start_date = ~D[2023-01-01]
      dates = Enum.map(0..(n_points - 1), fn i -> Date.add(start_date, i) end)
      y_values = Enum.map(1..n_points, fn i -> i * 1.0 + :rand.normal(0, 1) end)

      df = Explorer.DataFrame.new(%{"ds" => dates, "y" => y_values})

      model =
        Soothsayer.new(%{
          trend: %{enabled: false, changepoints: 0},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          ar: %{enabled: true, lags: 3},
          epochs: 2
        })

      fitted_model = Soothsayer.fit(model, df)
      weights = AR.get_weights(fitted_model)

      assert Map.has_key?(weights, "ar_dense_out")
      assert map_size(weights) == 1
      assert Nx.shape(weights["ar_dense_out"].kernel) == {3, 1}
      assert Nx.shape(weights["ar_dense_out"].bias) == {1}
    end

    test "returns all layer weights for deep AR-Net" do
      :rand.seed(:exsss, {42, 42, 42})

      n_points = 50
      start_date = ~D[2023-01-01]
      dates = Enum.map(0..(n_points - 1), fn i -> Date.add(start_date, i) end)
      y_values = Enum.map(1..n_points, fn i -> i * 1.0 + :rand.normal(0, 1) end)

      df = Explorer.DataFrame.new(%{"ds" => dates, "y" => y_values})

      model =
        Soothsayer.new(%{
          trend: %{enabled: false, changepoints: 0},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          ar: %{enabled: true, lags: 5, layers: [16, 8]},
          epochs: 2
        })

      fitted_model = Soothsayer.fit(model, df)
      weights = AR.get_weights(fitted_model)

      assert Map.has_key?(weights, "ar_dense_0")
      assert Map.has_key?(weights, "ar_dense_1")
      assert Map.has_key?(weights, "ar_dense_out")
      assert Nx.shape(weights["ar_dense_0"].kernel) == {5, 16}
      assert Nx.shape(weights["ar_dense_1"].kernel) == {16, 8}
      assert Nx.shape(weights["ar_dense_out"].kernel) == {8, 1}
    end

    test "raises error when AR is disabled" do
      model = Soothsayer.new(%{ar: %{enabled: false}})

      assert_raise ArgumentError, ~r/AR is not enabled/, fn ->
        AR.get_weights(model)
      end
    end

    test "raises error when model is not fitted" do
      model = Soothsayer.new(%{ar: %{enabled: true, lags: 3}})

      assert_raise ArgumentError, ~r/not been fitted/, fn ->
        AR.get_weights(model)
      end
    end
  end
end
