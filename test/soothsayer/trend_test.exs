defmodule Soothsayer.TrendTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Trend

  describe "build_input/1" do
    test "returns input with shape {nil, 1, 1} when changepoints is 0" do
      config = %{trend: %{changepoints: 0}}

      input = Trend.build_input(config)

      assert Axon.get_inputs(input)["trend"] == {nil, 1, 1}
    end

    test "returns input with shape {nil, positions, 1 + changepoints}" do
      config = %{trend: %{changepoints: 5}}

      input = Trend.build_input(config)

      assert Axon.get_inputs(input)["trend"] == {nil, 1, 6}
    end

    test "returns input with shape {nil, 1, 1} when trend config missing changepoints" do
      config = %{trend: %{}}

      input = Trend.build_input(config)

      assert Axon.get_inputs(input)["trend"] == {nil, 1, 1}
    end
  end

  describe "build_component/2" do
    test "returns dense layer when enabled" do
      config = %{trend: %{enabled: true, changepoints: 0}}
      input = Axon.input("trend", shape: {nil, 1, 1})

      component = Trend.build_component(input, config)

      # Build and check that it produces output
      {init_fn, _predict_fn} = Axon.build(component)
      params = init_fn.(%{"trend" => Nx.tensor([[[1.0]]])}, Axon.ModelState.empty())

      assert Map.has_key?(params.data, "trend_dense")
    end

    test "returns constant 0 when disabled" do
      config = %{trend: %{enabled: false, changepoints: 0}}
      input = Axon.input("trend", shape: {nil, 1, 1})

      component = Trend.build_component(input, config)

      # Build and check output is zeros
      {init_fn, predict_fn} = Axon.build(component)
      params = init_fn.(%{"trend" => Nx.tensor([[[1.0]]])}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"trend" => Nx.tensor([[[1.0]]])})

      assert Nx.to_number(output) == 0.0
    end

    test "dense layer is named 'trend_dense' for regularization" do
      config = %{trend: %{enabled: true, changepoints: 3}}
      input = Axon.input("trend", shape: {nil, 1, 4})

      component = Trend.build_component(input, config)

      {init_fn, _predict_fn} = Axon.build(component)

      params =
        init_fn.(%{"trend" => Nx.tensor([[[1.0, 0.0, 0.0, 0.0]]])}, Axon.ModelState.empty())

      assert Map.has_key?(params.data, "trend_dense")
    end
  end

  describe "get_weights/1" do
    test "extracts trend_dense kernel and bias from fitted model" do
      # Create a minimal model with trend enabled
      config = %{
        trend: %{enabled: true, changepoints: 2, changepoints_range: 0.8, regularization: nil},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: false, lags: 0}
      }

      model = Soothsayer.Model.new(config)

      # Initialize with dummy data
      input = %{
        "trend" => Nx.tensor([[[1.0, 0.0, 0.0]]]),
        "yearly" => Nx.broadcast(0.0, {1, 1, 8}),
        "weekly" => Nx.broadcast(0.0, {1, 1, 4})
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
        trend: %{enabled: true, changepoints: 0, changepoints_range: 0.8, regularization: nil},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: false, lags: 0}
      }

      model = Soothsayer.Model.new(config)

      assert_raise ArgumentError, ~r/not been fitted/, fn ->
        Trend.get_weights(model)
      end
    end

    test "raises when trend not enabled" do
      config = %{
        trend: %{enabled: false, changepoints: 0, changepoints_range: 0.8, regularization: nil},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: false, lags: 0}
      }

      model = Soothsayer.Model.new(config)

      # Initialize with dummy data
      input = %{
        "trend" => Nx.tensor([[[1.0]]]),
        "yearly" => Nx.broadcast(0.0, {1, 1, 8}),
        "weekly" => Nx.broadcast(0.0, {1, 1, 4})
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
    test "returns empty list when changepoints is 0" do
      result = Trend.compute_changepoint_indices(100, 0, 0.8)
      assert result == []
    end

    test "returns evenly spaced indices in the first portion of data" do
      result = Trend.compute_changepoint_indices(100, 5, 0.8)

      assert length(result) == 5
      assert Enum.all?(result, fn idx -> idx >= 0 and idx <= 80 end)
    end
  end

  describe "logistic growth" do
    test "capacity columns follow the growth and the floor the training data had" do
      assert Trend.capacity_columns(%{trend: %{growth: :linear}}) == []
      assert Trend.capacity_columns(%{trend: %{growth: :logistic}}) == ["cap"]

      assert Trend.capacity_columns(%{trend: %{growth: :logistic, uses_floor: true}}) == [
               "cap",
               "floor"
             ]
    end

    test "saturate squashes the trend between floor and cap" do
      config = %{
        trend: %{growth: :logistic, enabled: true, changepoints: 0},
        ar: %{enabled: false}
      }

      input = Trend.build_input(config)
      trend = Trend.build_component(input, config)

      {init_fn, predict_fn} =
        trend |> Trend.saturate(config) |> then(&Axon.container(%{trend: &1})) |> Axon.build()

      x = %{
        "trend" => Nx.tensor([[[0.0]], [[100.0]], [[-100.0]]]),
        "capacity" => Nx.tensor([[[2.0, -1.0]], [[2.0, -1.0]], [[2.0, -1.0]]])
      }

      params = init_fn.(x, Axon.ModelState.empty())
      params = put_in(params.data["trend_dense"]["bias"], Nx.tensor([0.0]))
      params = put_in(params.data["trend_dense"]["kernel"], Nx.tensor([[1.0]]))

      out = predict_fn.(params, x).trend |> Nx.to_flat_list()
      # sigmoid(0) = 0.5 of the way from -1 to 2, then the cap and the floor
      assert_in_delta Enum.at(out, 0), 0.5, 1.0e-6
      assert_in_delta Enum.at(out, 1), 2.0, 1.0e-6
      assert_in_delta Enum.at(out, 2), -1.0, 1.0e-6
    end
  end

  describe "changepoint_metadata/2 with dates" do
    test "puts the changepoints at the dates, in days from the first timestamp" do
      dates = Enum.map(0..99, fn i -> Date.add(~D[2023-01-01], i) end)
      config = %{trend: %{changepoints: [~D[2023-01-11], ~D[2023-03-01]]}}

      metadata = Trend.changepoint_metadata(dates, config)

      assert metadata.first_timestamp == ~D[2023-01-01]
      assert metadata.changepoint_positions == [10.0, 59.0]
      assert Trend.count(config) == 2
      assert Trend.feature_count(config) == 3
    end

    test "rejects a date outside the training span" do
      dates = Enum.map(0..99, fn i -> Date.add(~D[2023-01-01], i) end)

      assert_raise ArgumentError, ~r/outside the training data/, fn ->
        Trend.changepoint_metadata(dates, %{trend: %{changepoints: [~D[2023-06-01]]}})
      end

      assert_raise ArgumentError, ~r/outside the training data/, fn ->
        Trend.changepoint_metadata(dates, %{trend: %{changepoints: [~D[2023-01-01]]}})
      end
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

    test "segmentwise hinges stop at the next changepoint and the last one never does" do
      t = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

      result = Trend.build_changepoint_features(t, [1.5, 3.0], :segmentwise)

      assert Nx.to_flat_list(result) == [0.0, 0.0, 0.5, 0.0, 1.5, 0.0, 1.5, 1.0, 1.5, 2.0]
    end

    test "discontinuous growth appends one intercept column per changepoint" do
      t = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

      segmentwise = Trend.build_changepoint_features(t, [1.5, 3.0], :segmentwise, :discontinuous)
      assert Nx.shape(segmentwise) == {5, 4}
      # ramps live inside their segment, indicators are one-hot per segment
      assert Nx.to_list(segmentwise) == [
               [0.0, 0.0, 0.0, 0.0],
               [0.5, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0],
               [0.0, 1.0, 0.0, 1.0],
               [0.0, 2.0, 0.0, 1.0]
             ]

      cumulative = Trend.build_changepoint_features(t, [1.5, 3.0], :cumulative, :discontinuous)
      # hinges keep growing, indicators are steps
      assert Nx.to_list(cumulative) == [
               [0.0, 0.0, 0.0, 0.0],
               [0.5, 0.0, 1.0, 0.0],
               [1.5, 0.0, 1.0, 1.0],
               [2.5, 1.0, 1.0, 1.0],
               [3.5, 2.0, 1.0, 1.0]
             ]

      assert Trend.feature_count(%{trend: %{changepoints: 2, growth: :discontinuous}}) == 5
      assert Trend.feature_count(%{trend: %{changepoints: 0, growth: :discontinuous}}) == 1
      assert Trend.time_columns(%{trend: %{changepoints: 2, growth: :discontinuous}}) == 3
    end

    test "the basis follows trend regularization" do
      assert Trend.basis(%{trend: %{regularization: nil}}) == :segmentwise
      assert Trend.basis(%{trend: %{regularization: 0.5}}) == :cumulative
    end
  end

  describe "build_features/2" do
    test "returns tensor and metadata" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04], ~D[2023-01-05]]
      config = %{trend: %{changepoints: 2, changepoints_range: 0.8}}

      {tensor, metadata} = Trend.build_features(dates, config)

      assert is_struct(tensor, Nx.Tensor)
      assert Map.has_key?(metadata, :first_timestamp)
      assert Map.has_key?(metadata, :changepoint_positions)
    end

    test "tensor has shape {n_dates, 1 + changepoints}" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04], ~D[2023-01-05]]
      config = %{trend: %{changepoints: 2, changepoints_range: 0.8}}

      {tensor, _metadata} = Trend.build_features(dates, config)

      # 5 dates, 1 + 2 changepoints = 3 columns
      assert Nx.shape(tensor) == {5, 3}
    end

    test "first column is days since first date" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]
      config = %{trend: %{changepoints: 0, changepoints_range: 0.8}}

      {tensor, _metadata} = Trend.build_features(dates, config)

      # Days: 0, 1, 2
      first_col = Nx.slice(tensor, [0, 0], [3, 1]) |> Nx.flatten() |> Nx.to_flat_list()
      assert first_col == [0.0, 1.0, 2.0]
    end

    test "metadata includes first_timestamp" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]
      config = %{trend: %{changepoints: 0, changepoints_range: 0.8}}

      {_tensor, metadata} = Trend.build_features(dates, config)

      assert metadata.first_timestamp == ~D[2023-01-01]
    end

    test "metadata includes changepoint_positions as numeric values" do
      dates = Enum.map(0..99, fn i -> Date.add(~D[2023-01-01], i) end)
      config = %{trend: %{changepoints: 5, changepoints_range: 0.8}}

      {_tensor, metadata} = Trend.build_features(dates, config)

      assert is_list(metadata.changepoint_positions)
      assert length(metadata.changepoint_positions) == 5
      assert Enum.all?(metadata.changepoint_positions, &is_number/1)
    end

    test "handles zero changepoints" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]
      config = %{trend: %{changepoints: 0, changepoints_range: 0.8}}

      {tensor, metadata} = Trend.build_features(dates, config)

      assert Nx.shape(tensor) == {3, 1}
      assert metadata.changepoint_positions == []
    end
  end
end
