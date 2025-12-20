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

  describe "build_input/3" do
    test "builds AR input tensor from training data and dates" do
      training_data = %{
        dates: [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04], ~D[2023-01-05]],
        y_normalized: [1.0, 2.0, 3.0, 4.0, 5.0]
      }

      lags = 2
      prediction_dates = [~D[2023-01-04], ~D[2023-01-05]]

      result = AR.build_input(training_data, prediction_dates, lags)

      # For date 2023-01-04 (idx 3), lags are [2.0, 3.0]
      # For date 2023-01-05 (idx 4), lags are [3.0, 4.0]
      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [2.0, 3.0, 3.0, 4.0]
    end

    test "returns zeros for dates at beginning of series" do
      training_data = %{
        dates: [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]],
        y_normalized: [1.0, 2.0, 3.0]
      }

      lags = 2
      prediction_dates = [~D[2023-01-01], ~D[2023-01-02]]

      result = AR.build_input(training_data, prediction_dates, lags)

      # Both dates don't have enough history, should return zeros
      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.0, 0.0]
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
