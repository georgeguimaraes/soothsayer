defmodule Soothsayer.NeuralProphetBenchmarkTest do
  @moduledoc """
  Fits Soothsayer on the datasets NeuralProphet uses in its own model
  performance suite (`tests/test_model_performance.py`) and reports validation
  metrics next to the numbers NeuralProphet publishes for the same splits.

  Excluded from the default run. Execute with:

      mix test --only benchmark

  The NeuralProphet reference values come from the "Model Benchmark" comment
  its CI posted on https://github.com/ourownstory/neural_prophet/pull/1649.
  Where Soothsayer can't match NeuralProphet's configuration yet, the notes
  column says what differs, so the comparison is a parity report, not a
  pass/fail. The assertions are regression ceilings set from Soothsayer's own
  measured results.
  """

  use ExUnit.Case, async: false

  @moduletag :benchmark

  alias Explorer.DataFrame
  alias Explorer.Series

  @fixtures Path.expand("../fixtures/neuralprophet", __DIR__)
  @validation_fraction 0.1
  @seed 42

  setup_all do
    IO.puts("\n| Benchmark | Metric | NeuralProphet | Soothsayer | Ratio | Notes |")
    IO.puts("|---|---|---|---|---|---|")
    :ok
  end

  describe "Peyton Manning (daily log page views)" do
    test "default configuration matches NeuralProphet's defaults" do
      {train, validation} = load_and_split("wp_log_peyton_manning.csv")

      fitted_model = Soothsayer.fit(Soothsayer.new(%{seed: @seed}), train)
      predictions = Soothsayer.predict(fitted_model, validation["ds"])

      metrics = validation_metrics(predictions, validation["y"])

      report(
        "PeytonManning",
        metrics,
        %{mean_absolute_error: 0.35033, root_mean_squared_error: 0.50095},
        notes: "same config: 10 changepoints, yearly 6, weekly 3, additive"
      )

      # Seed 42 gives 0.286 / 0.473. Across six seeds: MAE 0.286 to 0.354,
      # RMSE 0.473 to 0.535. Ceilings are 1.25x the worst seed.
      assert metrics.mean_absolute_error < 0.45
      assert metrics.root_mean_squared_error < 0.67
    end
  end

  describe "Air Passengers (monthly)" do
    test "multiplicative yearly seasonality on monthly data" do
      {train, validation} = load_and_split("air_passengers.csv")

      model =
        Soothsayer.new(%{
          seasonality: %{mode: :multiplicative, weekly: %{enabled: false}},
          seed: @seed
        })

      fitted_model = Soothsayer.fit(model, train)
      predictions = Soothsayer.predict(fitted_model, validation["ds"])

      metrics = validation_metrics(predictions, validation["y"])

      report(
        "AirPassengers",
        metrics,
        %{mean_absolute_error: 30.1315, root_mean_squared_error: 31.0835},
        notes: "same config: multiplicative seasonality, weekly disabled for monthly rows"
      )

      # Seed 42 gives 23.1 / 25.0. Across six seeds: MAE 22.0 to 30.7,
      # RMSE 24.1 to 33.2 (additive mode was 31 to 32 / 38 to 42). Only 130
      # training rows, so init matters a lot here. Ceilings are 1.25x the worst seed.
      assert metrics.mean_absolute_error < 38.5
      assert metrics.root_mean_squared_error < 41.5
    end
  end

  describe "Energy price daily" do
    test "auto-regression with 14 lags and temperature as a future regressor, one step ahead" do
      {train, validation} = load_and_split("energy_price_daily.csv")

      model =
        Soothsayer.new(%{
          ar: %{enabled: true, lags: 14},
          trend: %{changepoints: 0},
          regressors: ["temperature"],
          seed: @seed
        })

      fitted_model = Soothsayer.fit(model, train)

      # Validation actuals seed the lags, so every prediction is one step
      # ahead from observed values, the same footing as NeuralProphet's
      # validation metrics. Temperature is known for the validation dates.
      predictions =
        Soothsayer.predict(fitted_model, validation["ds"],
          history: validation,
          regressors: validation
        )

      metrics = validation_metrics(predictions, validation["y"])

      report(
        "EnergyPriceDaily",
        metrics,
        %{mean_absolute_error: 5.40186, root_mean_squared_error: 6.70655},
        notes:
          "one step ahead vs NeuralProphet's 7-step average; NeuralProphet also lagged temperature"
      )

      # Seed 42 gives 4.96 / 6.31. Across six seeds: MAE 4.80 to 5.28,
      # RMSE 6.06 to 6.71. Without temperature it was 4.66 to 5.19 / 5.97
      # to 6.59: one step ahead, 14 lags of price already carry the weather.
      # Ceilings are 1.25x the worst seed.
      assert metrics.mean_absolute_error < 6.6
      assert metrics.root_mean_squared_error < 8.4
    end
  end

  # Loads a fixture and splits it like NeuralProphet's `split_df(valid_p: 0.1)`
  # does for a model without lags: the last `trunc(n * 0.1)` rows are validation.
  defp load_and_split(file) do
    dataframe =
      @fixtures
      |> Path.join(file)
      |> DataFrame.from_csv!(dtypes: [{"ds", :date}])

    row_count = DataFrame.n_rows(dataframe)
    validation_rows = max(1, trunc(row_count * @validation_fraction))
    training_rows = row_count - validation_rows

    {DataFrame.head(dataframe, training_rows), DataFrame.tail(dataframe, validation_rows)}
  end

  defp validation_metrics(predictions, actual) do
    predicted = Nx.flatten(predictions)
    observed = actual |> Series.to_tensor() |> Nx.as_type({:f, 32})
    errors = Nx.subtract(predicted, observed)

    %{
      mean_absolute_error: errors |> Nx.abs() |> Nx.mean() |> Nx.to_number(),
      root_mean_squared_error: errors |> Nx.pow(2) |> Nx.mean() |> Nx.sqrt() |> Nx.to_number()
    }
  end

  defp report(benchmark, metrics, reference, notes: notes) do
    for {metric, label} <- [mean_absolute_error: "MAE", root_mean_squared_error: "RMSE"] do
      ours = metrics[metric]
      theirs = reference[metric]

      IO.puts(
        "| #{benchmark} | #{label} | #{format(theirs)} | #{format(ours)} | #{Float.round(ours / theirs, 2)}x | #{notes} |"
      )
    end
  end

  defp format(value), do: :erlang.float_to_binary(value * 1.0, decimals: 4)
end
