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

  setup_all do
    IO.puts("\n| Benchmark | Metric | NeuralProphet | Soothsayer | Ratio | Notes |")
    IO.puts("|---|---|---|---|---|---|")
    :ok
  end

  describe "Peyton Manning (daily log page views)" do
    test "default configuration matches NeuralProphet's defaults" do
      :rand.seed(:exsss, {1, 2, 3})
      {train, validation} = load_and_split("wp_log_peyton_manning.csv")

      fitted_model = Soothsayer.fit(Soothsayer.new(), train)
      predictions = Soothsayer.predict(fitted_model, validation["ds"])

      metrics = validation_metrics(predictions, validation["y"])

      report(
        "PeytonManning",
        metrics,
        %{mean_absolute_error: 0.35033, root_mean_squared_error: 0.50095},
        notes: "same config: 10 changepoints, yearly 6, weekly 3, additive"
      )

      # Measured 0.304 / 0.499 over three runs, ceilings are 1.25x the worst run
      assert metrics.mean_absolute_error < 0.38
      assert metrics.root_mean_squared_error < 0.62
    end
  end

  describe "Air Passengers (monthly)" do
    test "additive yearly seasonality on monthly data" do
      :rand.seed(:exsss, {1, 2, 3})
      {train, validation} = load_and_split("air_passengers.csv")

      model = Soothsayer.new(%{seasonality: %{weekly: %{enabled: false}}})
      fitted_model = Soothsayer.fit(model, train)
      predictions = Soothsayer.predict(fitted_model, validation["ds"])

      metrics = validation_metrics(predictions, validation["y"])

      report(
        "AirPassengers",
        metrics,
        %{mean_absolute_error: 30.1315, root_mean_squared_error: 31.0835},
        notes: "NeuralProphet used multiplicative seasonality, Soothsayer is additive only"
      )

      # Measured 31.0 to 32.1 / 38.5 to 41.8 over three runs, only 130 training rows
      assert metrics.mean_absolute_error < 40.0
      assert metrics.root_mean_squared_error < 52.0
    end
  end

  describe "Energy price daily" do
    test "auto-regression with 14 lags, one step ahead" do
      :rand.seed(:exsss, {1, 2, 3})
      {train, validation} = load_and_split("energy_price_daily.csv")

      model = Soothsayer.new(%{ar: %{enabled: true, lags: 14}, trend: %{changepoints: 0}})
      fitted_model = Soothsayer.fit(model, train)

      # Validation actuals seed the lags, so every prediction is one step
      # ahead from observed values, the same footing as NeuralProphet's
      # validation metrics.
      predictions = Soothsayer.predict(fitted_model, validation["ds"], history: validation)

      metrics = validation_metrics(predictions, validation["y"])

      report(
        "EnergyPriceDaily",
        metrics,
        %{mean_absolute_error: 5.40186, root_mean_squared_error: 6.70655},
        notes:
          "NeuralProphet averaged 7 forecast steps and used temperature as a lagged and future regressor"
      )

      # Measured 4.729 / 6.034 over three runs
      assert metrics.mean_absolute_error < 5.9
      assert metrics.root_mean_squared_error < 7.55
    end
  end

  # Loads a fixture and splits it like NeuralProphet's `split_df(valid_p: 0.1)`
  # does for a model without lags: the last `trunc(n * 0.1)` rows are validation.
  defp load_and_split(file) do
    dataframe =
      @fixtures
      |> Path.join(file)
      |> DataFrame.from_csv!(dtypes: [{"ds", :date}])
      |> DataFrame.select(["ds", "y"])

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
