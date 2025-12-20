defmodule Soothsayer.TrendTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Trend

  describe "build_input/1" do
    test "returns input with shape {nil, 1} when n_changepoints is 0" do
      config = %{trend: %{n_changepoints: 0}}

      input = Trend.build_input(config)

      assert Axon.get_inputs(input)["trend"] == {nil, 1}
    end

    test "returns input with shape {nil, 1 + n_changepoints}" do
      config = %{trend: %{n_changepoints: 5}}

      input = Trend.build_input(config)

      assert Axon.get_inputs(input)["trend"] == {nil, 6}
    end

    test "returns input with shape {nil, 1} when trend config missing n_changepoints" do
      config = %{trend: %{}}

      input = Trend.build_input(config)

      assert Axon.get_inputs(input)["trend"] == {nil, 1}
    end
  end

  describe "build_component/2" do
    test "returns dense layer when enabled" do
      config = %{trend: %{enabled: true, n_changepoints: 0}}
      input = Axon.input("trend", shape: {nil, 1})

      component = Trend.build_component(input, config)

      # Build and check that it produces output
      {init_fn, _predict_fn} = Axon.build(component)
      params = init_fn.(%{"trend" => Nx.tensor([[1.0]])}, Axon.ModelState.empty())

      assert Map.has_key?(params.data, "trend_dense")
    end

    test "returns constant 0 when disabled" do
      config = %{trend: %{enabled: false, n_changepoints: 0}}
      input = Axon.input("trend", shape: {nil, 1})

      component = Trend.build_component(input, config)

      # Build and check output is zeros
      {init_fn, predict_fn} = Axon.build(component)
      params = init_fn.(%{"trend" => Nx.tensor([[1.0]])}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"trend" => Nx.tensor([[1.0]])})

      assert Nx.to_number(output) == 0.0
    end

    test "dense layer is named 'trend_dense' for regularization" do
      config = %{trend: %{enabled: true, n_changepoints: 3}}
      input = Axon.input("trend", shape: {nil, 4})

      component = Trend.build_component(input, config)

      {init_fn, _predict_fn} = Axon.build(component)
      params = init_fn.(%{"trend" => Nx.tensor([[1.0, 0.0, 0.0, 0.0]])}, Axon.ModelState.empty())

      assert Map.has_key?(params.data, "trend_dense")
    end
  end

  describe "get_weights/1" do
    test "extracts trend_dense kernel and bias from fitted model" do
      # Create a minimal model with trend enabled
      config = %{
        trend: %{enabled: true, n_changepoints: 2, changepoints_range: 0.8, regularization: nil},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: false, n_lags: 0}
      }

      model = Soothsayer.Model.new(config)

      # Initialize with dummy data
      input = %{
        "trend" => Nx.tensor([[1.0, 0.0, 0.0]]),
        "yearly" => Nx.broadcast(0.0, {1, 8}),
        "weekly" => Nx.broadcast(0.0, {1, 4})
      }

      {init_fn, _predict_fn} = Axon.build(model.network)
      params = init_fn.(input, Axon.ModelState.empty())
      fitted_model = %{model | params: params}

      weights = Trend.get_weights(fitted_model)

      assert Map.has_key?(weights, :kernel)
      assert Map.has_key?(weights, :bias)
      # Kernel shape should be {3, 1} (1 + 2 changepoints -> 1 output)
      assert Nx.shape(weights.kernel) == {3, 1}
    end

    test "raises when model not fitted" do
      config = %{
        trend: %{enabled: true, n_changepoints: 0, changepoints_range: 0.8, regularization: nil},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: false, n_lags: 0}
      }

      model = Soothsayer.Model.new(config)

      assert_raise ArgumentError, ~r/not been fitted/, fn ->
        Trend.get_weights(model)
      end
    end

    test "raises when trend not enabled" do
      config = %{
        trend: %{enabled: false, n_changepoints: 0, changepoints_range: 0.8, regularization: nil},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: false, n_lags: 0}
      }

      model = Soothsayer.Model.new(config)

      # Initialize with dummy data
      input = %{
        "trend" => Nx.tensor([[1.0]]),
        "yearly" => Nx.broadcast(0.0, {1, 8}),
        "weekly" => Nx.broadcast(0.0, {1, 4})
      }

      {init_fn, _predict_fn} = Axon.build(model.network)
      params = init_fn.(input, Axon.ModelState.empty())
      fitted_model = %{model | params: params}

      assert_raise ArgumentError, ~r/not enabled/, fn ->
        Trend.get_weights(fitted_model)
      end
    end
  end

  # Existing changepoint tests - these should still pass after rename
  describe "compute_changepoint_indices/3" do
    test "returns empty list when n_changepoints is 0" do
      result = Trend.compute_changepoint_indices(100, 0, 0.8)
      assert result == []
    end

    test "returns evenly spaced indices in the first portion of data" do
      result = Trend.compute_changepoint_indices(100, 5, 0.8)

      assert length(result) == 5
      assert Enum.all?(result, fn idx -> idx >= 0 and idx <= 80 end)
    end
  end

  describe "build_changepoint_features/2" do
    test "returns nil when no changepoint positions" do
      t = Nx.tensor([[1.0], [2.0], [3.0]])

      result = Trend.build_changepoint_features(t, [])

      assert result == nil
    end

    test "computes max(0, t - s_j) for each changepoint" do
      t = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      changepoint_positions = [2.5]

      result = Trend.build_changepoint_features(t, changepoint_positions)

      assert Nx.shape(result) == {5, 1}
      expected = Nx.tensor([[0.0], [0.0], [0.5], [1.5], [2.5]])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end
  end
end
