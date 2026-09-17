defmodule Soothsayer.NeuralProphetBenchmarkTest do
  @moduledoc """
  Fits Soothsayer on the datasets NeuralProphet uses in its own model
  performance suite (`tests/test_model_performance.py`) and reports validation
  metrics next to the numbers NeuralProphet publishes for the same splits.

  Excluded from the default run. Execute with:

      mix test --only benchmark

  The NeuralProphet reference values for the first four datasets come from
  the "Model Benchmark" comment its CI posted on
  https://github.com/ourownstory/neural_prophet/pull/1649. The R page views,
  US births and pedestrian panel references were measured on 2026-09-17 by
  running NeuralProphet 1.0.0rc10 (commit 5e6b23145473) on the same split
  with `split_df(valid_p=0.1)` and `fit(validation_df)`, seed 42; the
  panel's metrics were recomputed in absolute units since NeuralProphet
  reports a panel in normalized ones.
  Prophet's numbers come from Prophet 1.4.0 fitted on the same training rows
  with the same seasonality settings and predicting the same validation rows
  (`prophet_refs.py` in the session notes); it has no auto-regression, so on
  Yosemite and Energy it forecasts from time features alone and the number
  says what that costs.
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
    IO.puts(
      "\n| Benchmark | Metric | Prophet | NeuralProphet | Soothsayer | vs NeuralProphet | Notes |"
    )

    IO.puts("|---|---|---|---|---|---|---|")
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
        prophet: %{mean_absolute_error: 0.2915, root_mean_squared_error: 0.4774},
        notes: "same config: 10 changepoints, yearly 6, weekly 3, additive"
      )

      # Seed 42 gives 0.350 / 0.502, NeuralProphet's own number, since the
      # changepoints sit where NeuralProphet puts them (last one at 0.8 *
      # 10 / 11 of the data). Across six seeds: MAE 0.348 to 0.352, RMSE
      # 0.500 to 0.502. With the last changepoint at 0.8 exactly, Prophet's
      # convention, it was 0.297 / 0.490: the shorter tail segment suits this
      # series. Ceilings are 1.25x the worst seed.
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
        prophet: %{mean_absolute_error: 24.2298, root_mean_squared_error: 27.8352},
        notes: "same config: multiplicative seasonality, weekly disabled for monthly rows"
      )

      # Seed 42 gives 23.2 / 24.8 (24.7 / 26.3 with the last changepoint at
      # 0.8 exactly). Across six seeds: MAE 20.3 to 26.1, RMSE 22.1 to 27.6
      # (additive mode was 31 to 32 / 38 to 42). Only 130 training rows, so
      # init matters a lot here. Ceilings are 1.25x the worst seed.
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
        prophet: %{mean_absolute_error: 9.6374, root_mean_squared_error: 11.4534},
        notes:
          "same configuration and metric; Prophet has no lags, temperature as a regressor only"
      )

      # Seed 42 gives 5.42 / 6.75. Across six seeds: MAE 5.40 to 5.46,
      # RMSE 6.72 to 6.81 (with a fixed 0.01 rate and 100 epochs it was
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

      # One hour of readings (12 rows) is NaN on 2017-06-10. Fit imputes
      # them linearly, as NeuralProphet does.
      data = load("yosemite_temps.csv", {:naive_datetime, :microsecond})

      result = Soothsayer.backtest(model, data)

      assert result.model.config.frequency == {5, :minute}
      assert result.model.config.seasonality.daily.enabled

      report(
        "YosemiteTemps",
        result,
        %{mean_absolute_error: 0.57336, root_mean_squared_error: 0.84714},
        prophet: %{mean_absolute_error: 5.6292, root_mean_squared_error: 6.9022},
        notes:
          "same config: 36 lags, 12 steps, 30 changepoints, daily seasonality; Prophet has no lags; " <>
            "yearly off explicitly, NeuralProphet's auto rule turns it off on 65 days of data; " <>
            "12 missing readings imputed at fit"
      )

      # Seed 42 gives 0.467 / 0.689. Across six seeds: MAE 0.46 to 0.51,
      # RMSE 0.69 to 0.72. Before the AR saw stationarized lags it was 0.63
      # to 1.00 / 0.88 to 1.25. Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 0.70
      assert result.metrics.root_mean_squared_error < 0.95
    end
  end

  describe "R page views with outliers (daily log page views)" do
    test "default configuration on a series with outlier spikes" do
      result = Soothsayer.backtest(Soothsayer.new(%{seed: @seed}), load("wp_log_R_outliers.csv"))

      report(
        "RPageViewsOutliers",
        result,
        %{mean_absolute_error: 0.2271, root_mean_squared_error: 0.3219},
        prophet: %{mean_absolute_error: 0.2699, root_mean_squared_error: 0.3819},
        notes: "same config: defaults, the outlier spikes Prophet's docs use"
      )

      # Seed 42 gives 0.225 / 0.321. Across six seeds: MAE 0.225 to 0.226,
      # RMSE 0.320 to 0.321. With the last changepoint at 0.8 exactly it was
      # 0.399 / 0.486. Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 0.29
      assert result.metrics.root_mean_squared_error < 0.41
    end
  end

  describe "US births (daily)" do
    test "country holidays on twenty years of daily counts" do
      model = Soothsayer.new(%{holidays: %{countries: ["US"]}, seed: @seed})
      result = Soothsayer.backtest(model, load("births_us.csv"))

      report(
        "BirthsUS",
        result,
        %{mean_absolute_error: 446.9993, root_mean_squared_error: 532.3921},
        prophet: %{mean_absolute_error: 434.4995, root_mean_squared_error: 514.5654},
        notes: "same config: defaults plus US holidays (add_country_holidays)"
      )

      # Seed 42 gives 438 / 520. Across six seeds: MAE 438 to 442, RMSE 520
      # to 525. Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 555.0
      assert result.metrics.root_mean_squared_error < 660.0
    end
  end

  describe "Pedestrian counts at two locations (hourly panel)" do
    test "one model over two series with shared weekly and daily seasonality" do
      model =
        Soothsayer.new(%{
          series: %{column: "id"},
          seasonality: %{
            yearly: %{enabled: false},
            weekly: %{enabled: true},
            daily: %{enabled: true}
          },
          seed: @seed
        })

      data = load("pedestrians_panel.csv", {:naive_datetime, :microsecond})
      result = Soothsayer.backtest(model, data)

      assert result.model.config.series.ids == ["location_4", "location_41"]

      report(
        "PedestriansPanel",
        result,
        %{mean_absolute_error: 306.6183, root_mean_squared_error: 392.2146},
        prophet: %{mean_absolute_error: 339.1293, root_mean_squared_error: 432.4934},
        notes:
          "same config: ID column, global model, local normalization, weekly and daily " <>
            "seasonality, yearly off on one month of data"
      )

      # Seed 42 gives 303 / 386. Across six seeds: MAE 296 to 305, RMSE 380
      # to 388. Ceilings are 1.25x the worst seed.
      assert result.metrics.mean_absolute_error < 385.0
      assert result.metrics.root_mean_squared_error < 490.0
    end
  end

  defp load(file, ds_dtype \\ :date) do
    @fixtures |> Path.join(file) |> DataFrame.from_csv!(dtypes: [{"ds", ds_dtype}])
  end

  defp report(benchmark, result, reference, prophet: prophet, notes: notes) do
    config = result.model.config

    training =
      "lr #{:erlang.float_to_binary(config.learning_rate, [:short])}, #{config.epochs} epochs"

    for {metric, label} <- [mean_absolute_error: "MAE", root_mean_squared_error: "RMSE"] do
      ours = result.metrics[metric]
      theirs = reference[metric]

      IO.puts(
        "| #{benchmark} | #{label} | #{format(prophet[metric])} | #{format(theirs)} | " <>
          "#{format(ours)} | #{Float.round(ours / theirs, 2)}x | #{notes} (#{training}) |"
      )
    end
  end

  defp format(value), do: :erlang.float_to_binary(value * 1.0, decimals: 4)
end
