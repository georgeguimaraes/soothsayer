defmodule Soothsayer.ChangepointsTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Changepoints

  describe "config defaults" do
    test "new/1 includes trend changepoint config with defaults" do
      model = Soothsayer.new()

      assert model.config.trend.n_changepoints == 10
      assert model.config.trend.changepoints_range == 0.8
      assert model.config.trend.regularization == nil
    end

    test "n_changepoints: 0 disables changepoints" do
      model = Soothsayer.new(%{trend: %{n_changepoints: 0}})

      assert model.config.trend.n_changepoints == 0
    end
  end

  describe "network with changepoints" do
    test "build_network has trend input shape {nil, 1} when n_changepoints is 0" do
      config = %{
        trend: %{enabled: true, n_changepoints: 0, changepoints_range: 0.8, regularization: nil},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: false, n_lags: 0}
      }

      network = Soothsayer.Model.build_network(config)
      inputs = Axon.get_inputs(network)

      assert inputs["trend"] == {nil, 1}
    end

    test "build_network has trend input shape {nil, 1 + n_changepoints} when changepoints enabled" do
      config = %{
        trend: %{enabled: true, n_changepoints: 5, changepoints_range: 0.8, regularization: nil},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: false, n_lags: 0}
      }

      network = Soothsayer.Model.build_network(config)
      inputs = Axon.get_inputs(network)

      assert inputs["trend"] == {nil, 6}
    end

    test "build_network names trend layer 'trend_dense' for regularization" do
      config = %{
        trend: %{enabled: true, n_changepoints: 5, changepoints_range: 0.8, regularization: nil},
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 4},
          weekly: %{enabled: false, fourier_terms: 2}
        },
        ar: %{enabled: false, n_lags: 0}
      }

      network = Soothsayer.Model.build_network(config)
      {init_fn, _predict_fn} = Axon.build(network)

      input = %{
        "trend" => Nx.broadcast(0.0, {1, 6}),
        "yearly" => Nx.broadcast(0.0, {1, 8}),
        "weekly" => Nx.broadcast(0.0, {1, 4})
      }

      params = init_fn.(input, Axon.ModelState.empty())

      assert Map.has_key?(params.data, "trend_dense")
    end
  end

  describe "compute_changepoint_indices/3" do
    test "returns empty list when n_changepoints is 0" do
      result = Changepoints.compute_changepoint_indices(100, 0, 0.8)
      assert result == []
    end

    test "returns evenly spaced indices in the first portion of data" do
      # 100 samples, 5 changepoints, 80% range = first 80 samples
      # Changepoints at: 16, 32, 48, 64, 80 (evenly spaced)
      result = Changepoints.compute_changepoint_indices(100, 5, 0.8)

      assert length(result) == 5
      assert Enum.all?(result, fn idx -> idx >= 0 and idx <= 80 end)
      # Check that indices are evenly spaced
      [first | _rest] = result
      spacing = Enum.at(result, 1) - first

      assert Enum.all?(Enum.chunk_every(result, 2, 1, :discard), fn [a, b] ->
               b - a == spacing
             end)
    end

    test "respects changepoints_range parameter" do
      # 100 samples, 5 changepoints, 50% range = first 50 samples
      result = Changepoints.compute_changepoint_indices(100, 5, 0.5)

      assert length(result) == 5
      assert Enum.all?(result, fn idx -> idx >= 0 and idx <= 50 end)
    end

    test "handles edge case with few samples" do
      result = Changepoints.compute_changepoint_indices(10, 3, 0.8)

      assert length(result) == 3
      assert Enum.all?(result, fn idx -> idx >= 0 and idx <= 8 end)
    end
  end

  describe "compute_changepoint_positions/3" do
    test "returns dates at computed indices" do
      dates = Enum.map(0..99, fn i -> Date.add(~D[2023-01-01], i) end)

      result = Changepoints.compute_changepoint_positions(dates, 5, 0.8)

      assert length(result) == 5
      assert Enum.all?(result, fn date -> date in dates end)
      # First 80 days = up to index 80
      assert Enum.all?(result, fn date -> Date.compare(date, ~D[2023-03-23]) != :gt end)
    end

    test "returns empty list when n_changepoints is 0" do
      dates = Enum.map(0..99, fn i -> Date.add(~D[2023-01-01], i) end)

      result = Changepoints.compute_changepoint_positions(dates, 0, 0.8)

      assert result == []
    end
  end

  describe "build_changepoint_features/2" do
    test "returns nil when no changepoint positions" do
      t = Nx.tensor([[1.0], [2.0], [3.0]])

      result = Changepoints.build_changepoint_features(t, [])

      assert result == nil
    end

    test "computes max(0, t - s_j) for each changepoint" do
      # t values: 1, 2, 3, 4, 5
      # changepoint at t=2.5
      # Expected: max(0, 1-2.5)=0, max(0, 2-2.5)=0, max(0, 3-2.5)=0.5, max(0, 4-2.5)=1.5, max(0, 5-2.5)=2.5
      t = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      changepoint_positions = [2.5]

      result = Changepoints.build_changepoint_features(t, changepoint_positions)

      assert Nx.shape(result) == {5, 1}
      expected = Nx.tensor([[0.0], [0.0], [0.5], [1.5], [2.5]])
      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end

    test "handles multiple changepoints" do
      # t values: 1, 2, 3, 4, 5
      # changepoints at t=1.5 and t=3.5
      t = Nx.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
      changepoint_positions = [1.5, 3.5]

      result = Changepoints.build_changepoint_features(t, changepoint_positions)

      assert Nx.shape(result) == {5, 2}
      # Column 0: max(0, t-1.5) = [0, 0.5, 1.5, 2.5, 3.5]
      # Column 1: max(0, t-3.5) = [0, 0, 0, 0.5, 1.5]
      expected =
        Nx.tensor([
          [0.0, 0.0],
          [0.5, 0.0],
          [1.5, 0.0],
          [2.5, 0.5],
          [3.5, 1.5]
        ])

      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end
  end

  describe "build_trend_input/2" do
    test "returns t when no changepoints" do
      t = Nx.tensor([[1.0], [2.0], [3.0]])

      result = Changepoints.build_trend_input(t, nil)

      assert Nx.shape(result) == {3, 1}
      assert Nx.to_flat_list(result) == Nx.to_flat_list(t)
    end

    test "concatenates t with changepoint features" do
      t = Nx.tensor([[1.0], [2.0], [3.0]])
      changepoint_features = Nx.tensor([[0.0, 0.0], [0.5, 0.0], [1.5, 0.5]])

      result = Changepoints.build_trend_input(t, changepoint_features)

      assert Nx.shape(result) == {3, 3}

      expected =
        Nx.tensor([
          [1.0, 0.0, 0.0],
          [2.0, 0.5, 0.0],
          [3.0, 1.5, 0.5]
        ])

      assert Nx.to_flat_list(result) == Nx.to_flat_list(expected)
    end
  end

  describe "date_to_numeric/2" do
    test "converts dates to numeric values relative to first date" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]
      first_date = ~D[2023-01-01]

      result = Changepoints.date_to_numeric(dates, first_date)

      assert Nx.to_flat_list(result) == [0.0, 1.0, 2.0]
    end

    test "handles single date" do
      dates = [~D[2023-01-01]]
      first_date = ~D[2023-01-01]

      result = Changepoints.date_to_numeric(dates, first_date)

      assert Nx.to_flat_list(result) == [0.0]
    end
  end

  describe "numeric_to_date/2" do
    test "converts numeric values back to dates" do
      numeric = [0.0, 1.0, 2.0]
      first_date = ~D[2023-01-01]

      result = Changepoints.numeric_to_date(numeric, first_date)

      assert result == [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]
    end
  end

  describe "fit/predict integration" do
    alias Explorer.DataFrame
    alias Explorer.Series

    test "stores changepoint_positions in model config after fit" do
      :rand.seed(:exsss, {42, 42, 42})

      n_points = 100
      start_date = ~D[2023-01-01]
      dates = Enum.map(0..(n_points - 1), fn i -> Date.add(start_date, i) end)
      y_values = Enum.map(1..n_points, fn i -> i * 1.0 + :rand.normal(0, 1) end)

      df = DataFrame.new(%{"ds" => dates, "y" => y_values})

      model =
        Soothsayer.new(%{
          trend: %{n_changepoints: 5, changepoints_range: 0.8},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          epochs: 2
        })

      fitted_model = Soothsayer.fit(model, df)

      # Changepoint positions should be stored as numeric values
      assert Map.has_key?(fitted_model.config, :changepoint_positions)
      assert length(fitted_model.config.changepoint_positions) == 5
      # All positions should be numeric (days since first date)
      assert Enum.all?(fitted_model.config.changepoint_positions, &is_number/1)
    end

    test "fit and predict with changepoints enabled" do
      :rand.seed(:exsss, {42, 42, 42})

      n_points = 100
      start_date = ~D[2023-01-01]
      dates = Enum.map(0..(n_points - 1), fn i -> Date.add(start_date, i) end)
      # Generate data with a trend that changes slope
      y_values =
        Enum.map(0..(n_points - 1), fn i ->
          base = if i < 50, do: i * 1.0, else: 50.0 + (i - 50) * 2.0
          base + :rand.normal(0, 2)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y_values})

      model =
        Soothsayer.new(%{
          trend: %{n_changepoints: 5, changepoints_range: 0.8},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          epochs: 5
        })

      fitted_model = Soothsayer.fit(model, df)

      # Predict on last 10 dates
      test_dates = Enum.take(dates, -10) |> Series.from_list()
      predictions = Soothsayer.predict(fitted_model, test_dates)

      assert Nx.shape(predictions) == {10, 1}
      # Model should produce reasonable predictions (not all zeros)
      pred_values = Nx.to_flat_list(predictions)
      assert Enum.any?(pred_values, fn v -> abs(v) > 1.0 end)
    end

    test "n_changepoints: 0 works (backward compatible)" do
      :rand.seed(:exsss, {42, 42, 42})

      n_points = 50
      start_date = ~D[2023-01-01]
      dates = Enum.map(0..(n_points - 1), fn i -> Date.add(start_date, i) end)
      y_values = Enum.map(1..n_points, fn i -> i * 1.0 + :rand.normal(0, 1) end)

      df = DataFrame.new(%{"ds" => dates, "y" => y_values})

      model =
        Soothsayer.new(%{
          trend: %{n_changepoints: 0},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          epochs: 2
        })

      fitted_model = Soothsayer.fit(model, df)

      # Should have empty changepoint_positions
      assert fitted_model.config.changepoint_positions == []

      # Predict should still work
      test_dates = Enum.take(dates, -5) |> Series.from_list()
      predictions = Soothsayer.predict(fitted_model, test_dates)

      assert Nx.shape(predictions) == {5, 1}
    end

    test "stores first_date for prediction" do
      :rand.seed(:exsss, {42, 42, 42})

      n_points = 50
      start_date = ~D[2023-01-01]
      dates = Enum.map(0..(n_points - 1), fn i -> Date.add(start_date, i) end)
      y_values = Enum.map(1..n_points, fn i -> i * 1.0 + :rand.normal(0, 1) end)

      df = DataFrame.new(%{"ds" => dates, "y" => y_values})

      model =
        Soothsayer.new(%{
          trend: %{n_changepoints: 3},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          epochs: 2
        })

      fitted_model = Soothsayer.fit(model, df)

      assert fitted_model.config.first_date == start_date
    end
  end
end
