defmodule Soothsayer.ARTest do
  use ExUnit.Case, async: true

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Preprocessor

  describe "auto-regression config" do
    test "new/1 includes AR config with defaults" do
      model = Soothsayer.new()

      assert model.config.ar.enabled == false
      assert model.config.ar.n_lags == 0
      assert model.config.ar.layers == []
      assert model.config.ar.regularization == nil
    end
  end

  describe "lagged input preparation" do
    test "create_lagged_inputs/2 creates sliding windows from y values" do
      y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      n_lags = 3

      {lagged, targets} = Preprocessor.create_lagged_inputs(y, n_lags)

      # With n_lags=3 and 5 values, we get 2 windows:
      # Window 1: [1, 2, 3] -> target: 4
      # Window 2: [2, 3, 4] -> target: 5
      expected_lagged = Nx.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
      expected_targets = Nx.tensor([[4.0], [5.0]])

      assert Nx.shape(lagged) == {2, 3}
      assert Nx.shape(targets) == {2, 1}
      assert Nx.to_flat_list(lagged) == Nx.to_flat_list(expected_lagged)
      assert Nx.to_flat_list(targets) == Nx.to_flat_list(expected_targets)
    end
  end

  describe "AR network" do
    test "build_network/1 includes AR input when enabled" do
      config = %{
        trend: %{enabled: true},
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 4},
          weekly: %{enabled: true, fourier_terms: 2}
        },
        ar: %{enabled: true, n_lags: 5}
      }

      network = Soothsayer.Model.build_network(config)
      inputs = Axon.get_inputs(network)

      assert Map.has_key?(inputs, "ar")
    end

    test "build_network/1 does not include AR input when disabled" do
      config = %{
        trend: %{enabled: true},
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 4},
          weekly: %{enabled: true, fourier_terms: 2}
        },
        ar: %{enabled: false, n_lags: 0}
      }

      network = Soothsayer.Model.build_network(config)
      inputs = Axon.get_inputs(network)

      refute Map.has_key?(inputs, "ar")
    end

    test "build_network/1 builds deep AR-Net with hidden layers" do
      config = %{
        trend: %{enabled: false},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: true, n_lags: 5, layers: [32, 16]}
      }

      network = Soothsayer.Model.build_network(config)

      # Build and initialize the network to verify it works
      {init_fn, predict_fn} = Axon.build(network)

      input = %{
        "trend" => Nx.tensor([[1.0]]),
        "yearly" => Nx.broadcast(0.0, {1, 8}),
        "weekly" => Nx.broadcast(0.0, {1, 4}),
        "ar" => Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
      }

      params = init_fn.(input, Axon.ModelState.empty())

      # Verify the deep AR-Net has the expected layer structure
      # With layers: [32, 16], we should have:
      # - ar_dense_0: {5, 32} weights + {32} bias (5 inputs -> 32 hidden)
      # - ar_dense_1: {32, 16} weights + {16} bias (32 -> 16 hidden)
      # - ar_dense_out: {16, 1} weights + {1} bias (16 -> 1 output)
      assert Map.has_key?(params.data, "ar_dense_0")
      assert Map.has_key?(params.data, "ar_dense_1")
      assert Map.has_key?(params.data, "ar_dense_out")

      # Verify prediction works and output has correct shape
      result = predict_fn.(params, input)
      assert Nx.shape(result.combined) == {1, 1}
    end

    test "build_network/1 applies L1 regularization when regularization is set" do
      # L1 regularization should push weights towards zero during training
      # We verify by training the same model with and without regularization
      # using identical initial weights
      config = %{
        trend: %{enabled: false},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: true, n_lags: 3, layers: [], regularization: nil}
      }

      # Create training data
      input = %{
        "trend" => Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]),
        "yearly" => Nx.broadcast(0.0, {5, 8}),
        "weekly" => Nx.broadcast(0.0, {5, 4}),
        "ar" => Nx.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]])
      }
      target = Nx.tensor([[4.0], [5.0], [6.0], [7.0], [8.0]])

      # Build network and get initial params
      network = Soothsayer.Model.build_network(config)
      {init_fn, _} = Axon.build(network)
      initial_params = init_fn.(input, Axon.ModelState.empty())

      # Train model WITHOUT regularization using standard loop
      trained_no_reg =
        network
        |> Axon.Loop.trainer(
          &Axon.Losses.huber(&1, &2.combined, reduction: :mean),
          Polaris.Optimizers.adam(learning_rate: 0.1)
        )
        |> Axon.Loop.run(Stream.repeatedly(fn -> {input, target} end), initial_params,
          epochs: 100,
          iterations: 5,
          compiler: EXLA
        )

      # Train model WITH regularization using same initial params
      # We use the Model.fit with regularization which uses custom training loop
      model_with_reg = %Soothsayer.Model{
        network: network,
        config: %{learning_rate: 0.1, ar: %{regularization: 1.0}}
      }

      trained_reg = Soothsayer.Model.train_with_regularization_public(
        model_with_reg, input, target, 100, initial_params, 1.0
      )

      # Get the AR dense layer weights
      ar_weights_reg = trained_reg.data["ar_dense_out"]["kernel"]
      ar_weights_no_reg = trained_no_reg.data["ar_dense_out"]["kernel"]

      # L1 regularization should result in smaller weight magnitudes
      l1_norm_reg = ar_weights_reg |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      l1_norm_no_reg = ar_weights_no_reg |> Nx.abs() |> Nx.sum() |> Nx.to_number()

      # With L1 regularization, weights should be pushed toward zero
      assert l1_norm_reg < l1_norm_no_reg,
        "L1 regularization should reduce weight magnitudes. Got reg: #{l1_norm_reg}, no_reg: #{l1_norm_no_reg}"
    end
  end

  describe "AR training and prediction" do
    test "fit and predict with deep AR-Net learns non-linear pattern" do
      # Generate data with a non-linear autoregressive pattern
      :rand.seed(:exsss, {42, 42, 42})

      n_points = 200
      start_date = ~D[2023-01-01]

      # Non-linear pattern: y(t) = sin(y(t-1)/20) * 10 + noise
      {y_values, _} =
        Enum.reduce(1..n_points, {[50.0], 50.0}, fn _, {acc, prev} ->
          next = :math.sin(prev / 20) * 10 + 50 + :rand.normal(0, 2)
          {[next | acc], next}
        end)

      y_values = Enum.reverse(y_values) |> Enum.take(n_points)
      dates = Enum.map(0..(n_points - 1), fn i -> Date.add(start_date, i) end)

      df = DataFrame.new(%{"ds" => dates, "y" => y_values})

      # Create model with deep AR-Net
      model =
        Soothsayer.new(%{
          trend: %{enabled: false},
          seasonality: %{
            yearly: %{enabled: false},
            weekly: %{enabled: false}
          },
          ar: %{enabled: true, n_lags: 5, layers: [16, 8]},
          epochs: 5
        })

      # Fit the model
      fitted_model = Soothsayer.fit(model, df)

      # Make predictions
      test_dates = Enum.take(dates, -10) |> Series.from_list()
      predictions = Soothsayer.predict(fitted_model, test_dates)

      # Predictions should have correct shape
      assert Nx.shape(predictions) == {10, 1}

      # Model should produce non-trivial predictions
      pred_values = Nx.to_flat_list(predictions)
      assert Enum.any?(pred_values, fn v -> abs(v - 50) > 1.0 end)
    end

    test "fit and predict with AR enabled learns autoregressive pattern" do
      # Generate data with a simple autoregressive pattern: y(t) = 0.8 * y(t-1) + noise
      # This is an AR(1) process
      :rand.seed(:exsss, {42, 42, 42})

      n_points = 200
      start_date = ~D[2023-01-01]

      {y_values, _} =
        Enum.reduce(1..n_points, {[100.0], 100.0}, fn _, {acc, prev} ->
          next = 0.8 * prev + :rand.normal(0, 5)
          {[next | acc], next}
        end)

      y_values = Enum.reverse(y_values) |> Enum.take(n_points)
      dates = Enum.map(0..(n_points - 1), fn i -> Date.add(start_date, i) end)

      df = DataFrame.new(%{"ds" => dates, "y" => y_values})

      # Create model with AR enabled
      model =
        Soothsayer.new(%{
          trend: %{enabled: false},
          seasonality: %{
            yearly: %{enabled: false},
            weekly: %{enabled: false}
          },
          ar: %{enabled: true, n_lags: 3},
          epochs: 5
        })

      # Fit the model
      fitted_model = Soothsayer.fit(model, df)

      # Make predictions for the last few dates (in-sample)
      test_dates = Enum.take(dates, -10) |> Series.from_list()
      predictions = Soothsayer.predict(fitted_model, test_dates)

      # Predictions should be tensors with the right shape
      assert Nx.shape(predictions) == {10, 1}

      # The model should have learned something (predictions shouldn't be all zeros)
      pred_values = Nx.to_flat_list(predictions)
      assert Enum.any?(pred_values, fn v -> abs(v) > 1.0 end)
    end
  end
end
