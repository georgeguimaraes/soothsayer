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
      result =
        Soothsayer.backtest(Soothsayer.new(%{seed: @seed}), load("wp_log_peyton_manning.csv"))

      report(
        "PeytonManning",
        result.metrics,
        %{mean_absolute_error: 0.35033, root_mean_squared_error: 0.50095},
        notes: "same config: 10 changepoints, yearly 6, weekly 3, additive"
      )

      # Seed 42 gives 0.296 / 0.490. Across six seeds: MAE 0.296 to 0.300,
      # RMSE 0.487 to 0.496 (with a fixed 0.01 rate and 100 epochs it was
      # 0.286 to 0.354). Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 0.375
      assert result.metrics.root_mean_squared_error < 0.62
    end
  end

  describe "Air Passengers (monthly)" do
    test "multiplicative yearly seasonality on monthly data" do
      model =
        Soothsayer.new(%{
          seasonality: %{mode: :multiplicative, weekly: %{enabled: false}},
          seed: @seed
        })

      result = Soothsayer.backtest(model, load("air_passengers.csv"))

      report(
        "AirPassengers",
        result.metrics,
        %{mean_absolute_error: 30.1315, root_mean_squared_error: 31.0835},
        notes: "same config: multiplicative seasonality, weekly disabled for monthly rows"
      )

      # Seed 42 gives 26.2 / 28.1. Across six seeds: MAE 23.2 to 29.8,
      # RMSE 25.2 to 31.8 (additive mode was 31 to 32 / 38 to 42). Only 130
      # training rows, so init matters a lot here. Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 37.3
      assert result.metrics.root_mean_squared_error < 39.7
    end
  end

  describe "Energy price daily" do
    test "auto-regression with 14 lags, 7 direct forecast steps and temperature as a future and lagged regressor" do
      model =
        Soothsayer.new(%{
          ar: %{enabled: true, lags: 14, forecast_steps: 7},
          trend: %{changepoints: 0},
          regressors: ["temperature"],
          lagged_regressors: %{"temperature" => %{lags: 3}},
          seed: @seed
        })

      # NeuralProphet's validation metrics average horizons 1 to 7 for every
      # validation date, which is what the backtest computes.
      result = Soothsayer.backtest(model, load("energy_price_daily.csv"))

      report(
        "EnergyPriceDaily",
        result.metrics,
        %{mean_absolute_error: 5.40186, root_mean_squared_error: 6.70655},
        notes: "same configuration and metric"
      )

      # Seed 42 gives 5.44 / 6.76. Across six seeds: MAE 5.42 to 5.47,
      # RMSE 6.74 to 6.81 (with a fixed 0.01 rate and 100 epochs it was
      # 5.57 to 6.11). Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 6.85
      assert result.metrics.root_mean_squared_error < 8.5
    end
  end

  defp load(file) do
    @fixtures |> Path.join(file) |> DataFrame.from_csv!(dtypes: [{"ds", :date}])
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
