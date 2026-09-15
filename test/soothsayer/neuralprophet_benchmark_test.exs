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
    @energy_forecast_steps 7

    test "auto-regression with 14 lags, 7 direct forecast steps and temperature as a regressor" do
      {train, validation} = load_and_split("energy_price_daily.csv")

      model =
        Soothsayer.new(%{
          ar: %{enabled: true, lags: 14, forecast_steps: @energy_forecast_steps},
          trend: %{changepoints: 0},
          regressors: ["temperature"],
          seed: @seed
        })

      fitted_model = Soothsayer.fit(model, train)

      metrics = multi_step_validation_metrics(fitted_model, validation, @energy_forecast_steps)

      report(
        "EnergyPriceDaily",
        metrics,
        %{mean_absolute_error: 5.40186, root_mean_squared_error: 6.70655},
        notes: "same config and metric; NeuralProphet also lagged temperature"
      )

      # Seed 42 gives 5.56 / 6.93. Across six seeds: MAE 5.48 to 5.96,
      # RMSE 6.88 to 7.54. Ceilings are 1.25x the worst seed.
      assert metrics.mean_absolute_error < 7.45
      assert metrics.root_mean_squared_error < 9.45
    end
  end

  # NeuralProphet's validation MAE/RMSE average over every horizon 1..n_forecasts
  # for every validation target. Reproduced here by forecasting a block from
  # each origin in turn: the last training date, then each validation date,
  # with history truncated at the origin so the lags are real observations.
  defp multi_step_validation_metrics(fitted_model, validation, forecast_steps) do
    validation_rows = DataFrame.n_rows(validation)

    actual_by_date =
      Enum.zip(Series.to_list(validation["ds"]), Series.to_list(validation["y"])) |> Map.new()

    validation_dates = Series.to_list(validation["ds"])

    pairs =
      Enum.flat_map(0..(validation_rows - 1), fn origin_rows ->
        target_dates = Enum.slice(validation_dates, origin_rows, forecast_steps)

        history =
          if origin_rows == 0, do: [], else: [history: DataFrame.head(validation, origin_rows)]

        predictions =
          fitted_model
          |> Soothsayer.predict(
            Series.from_list(target_dates),
            [regressors: validation] ++ history
          )
          |> Nx.to_flat_list()

        Enum.zip(predictions, Enum.map(target_dates, &actual_by_date[&1]))
      end)

    errors = pairs |> Enum.map(fn {predicted, actual} -> predicted - actual end) |> Nx.tensor()

    %{
      mean_absolute_error: errors |> Nx.abs() |> Nx.mean() |> Nx.to_number(),
      root_mean_squared_error: errors |> Nx.pow(2) |> Nx.mean() |> Nx.sqrt() |> Nx.to_number()
    }
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
