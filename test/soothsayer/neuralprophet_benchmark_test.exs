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
        result,
        %{mean_absolute_error: 0.35033, root_mean_squared_error: 0.50095},
        notes: "same config: 10 changepoints, yearly 6, weekly 3, additive"
      )

      # Seed 42 gives 0.299 / 0.480. Across six seeds: MAE 0.298 to 0.311,
      # RMSE 0.480 to 0.483 (with a fixed 0.01 rate and 100 epochs it was
      # 0.286 to 0.354). Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 0.39
      assert result.metrics.root_mean_squared_error < 0.61
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
        result,
        %{mean_absolute_error: 30.1315, root_mean_squared_error: 31.0835},
        notes: "same config: multiplicative seasonality, weekly disabled for monthly rows"
      )

      # Seed 42 gives 27.1 / 29.0. Across six seeds: MAE 24.8 to 27.1,
      # RMSE 26.7 to 29.0 (additive mode was 31 to 32 / 38 to 42). Only 130
      # training rows, so init matters a lot here. Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 34.0
      assert result.metrics.root_mean_squared_error < 36.5
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
        result,
        %{mean_absolute_error: 5.40186, root_mean_squared_error: 6.70655},
        notes: "same configuration and metric"
      )

      # Seed 42 gives 5.38 / 6.72. Across six seeds: MAE 5.37 to 5.42,
      # RMSE 6.70 to 6.77 (with a fixed 0.01 rate and 100 epochs it was
      # 5.57 to 6.11). Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 6.8
      assert result.metrics.root_mean_squared_error < 8.5
    end
  end

  describe "Yosemite temperatures (every 5 minutes)" do
    test "auto-regression with 36 lags and 12 direct forecast steps on sub-daily data" do
      model =
        Soothsayer.new(%{
          ar: %{enabled: true, lags: 36, forecast_steps: 12},
          trend: %{changepoints: 30, changepoints_range: 0.9},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          seed: @seed
        })

      # One hour of readings (12 rows) is missing on 2017-06-10. NeuralProphet
      # imputes them before fitting, so the same is done here.
      data =
        "yosemite_temps.csv"
        |> load({:naive_datetime, :microsecond})
        |> interpolate_missing_targets()

      result = Soothsayer.backtest(model, data)

      assert result.model.config.frequency == {5, :minute}
      assert result.model.config.seasonality.daily.enabled

      report(
        "YosemiteTemps",
        result,
        %{mean_absolute_error: 0.57336, root_mean_squared_error: 0.84714},
        notes:
          "same config: 36 lags, 12 steps, 30 changepoints, daily seasonality; " <>
            "yearly off explicitly, NeuralProphet's auto rule turns it off on 65 days of data; " <>
            "12 missing readings linearly interpolated"
      )

      # Seed 42 gives 0.500 / 0.728. Across six seeds: MAE 0.49 to 0.55,
      # RMSE 0.72 to 0.76. Before the AR saw stationarized lags it was 0.63
      # to 1.00 / 0.88 to 1.25. Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 0.70
      assert result.metrics.root_mean_squared_error < 0.95
    end
  end

  defp load(file, ds_dtype \\ :date) do
    @fixtures |> Path.join(file) |> DataFrame.from_csv!(dtypes: [{"ds", ds_dtype}])
  end

  # Linear interpolation across runs of NaN, the way NeuralProphet's
  # impute_missing fills short gaps.
  defp interpolate_missing_targets(data) do
    values = data["y"] |> Explorer.Series.cast({:f, 64}) |> Explorer.Series.to_list()
    known = values |> Enum.with_index() |> Enum.reject(fn {value, _} -> value == :nan end)
    known_indices = Enum.map(known, &elem(&1, 1))
    known_values = Map.new(known, fn {value, index} -> {index, value} end)

    filled =
      Enum.with_index(values)
      |> Enum.map(fn
        {:nan, index} ->
          previous = known_indices |> Enum.filter(&(&1 < index)) |> Enum.max()
          next = known_indices |> Enum.filter(&(&1 > index)) |> Enum.min()
          fraction = (index - previous) / (next - previous)
          known_values[previous] + fraction * (known_values[next] - known_values[previous])

        {value, _index} ->
          value
      end)

    DataFrame.put(data, "y", Explorer.Series.from_list(filled))
  end

  defp report(benchmark, result, reference, notes: notes) do
    config = result.model.config

    training =
      "lr #{:erlang.float_to_binary(config.learning_rate, [:short])}, #{config.epochs} epochs"

    for {metric, label} <- [mean_absolute_error: "MAE", root_mean_squared_error: "RMSE"] do
      ours = result.metrics[metric]
      theirs = reference[metric]

      IO.puts(
        "| #{benchmark} | #{label} | #{format(theirs)} | #{format(ours)} | " <>
          "#{Float.round(ours / theirs, 2)}x | #{notes} (#{training}) |"
      )
    end
  end

  defp format(value), do: :erlang.float_to_binary(value * 1.0, decimals: 4)
end
