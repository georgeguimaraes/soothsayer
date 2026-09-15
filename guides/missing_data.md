# Missing Data

Real series have holes: a sensor that stopped for an hour, a day nobody logged, a NaN from a join. Soothsayer handles them at fit the way NeuralProphet does, so you can pass the data as it is and read the log to see what happened.

A value counts as missing when it is `nil` or NaN. A row counts as missing when its timestamp is absent from the frequency grid (see [The Basics](basics.md) for how the frequency is inferred).

## Without auto-regression

Trend, seasonality, events and regressors only look at timestamps, so a gap is harmless: the rows with a missing `y` are dropped and the rest is fitted as usual. Missing rows stay missing.

```elixir
df = DataFrame.new(%{
  "ds" => [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-04]],
  "y" => [10.0, nil, 12.0]
})

fitted = Soothsayer.fit(Soothsayer.new(), df)
# [info] Dropped 1 rows with missing y values
```

## With auto-regression

The lags are the previous steps of the series, so the data has to be complete at the model's frequency. Fit does three things in order:

1. **Regrid.** Every timestamp between the first and the last row that isn't in the data becomes a row of missing values. A timestamp that isn't on the grid at all (14:03 in a 5-minute series) raises, naming it.
2. **Drop the tail.** Rows at the end with a missing `y` are dropped rather than made up, so the series ends on something observed.
3. **Impute.** Every gap in `y`, in the regressor columns and in the lagged regressor columns is filled in two passes:
   - linearly between the gap's neighbours, up to `impute_linear` values from each end, so gaps of up to `2 * impute_linear` steps are filled completely;
   - then with a rolling mean over a centered window of `impute_rolling + 2 * impute_linear` steps, for whatever the first pass left open.

   Gaps longer than `2 * impute_linear + impute_rolling` steps (30 with the defaults) keep their middle.

```elixir
model = Soothsayer.new(%{ar: %{enabled: true, lags: 36, forecast_steps: 12}})
fitted = Soothsayer.fit(model, readings)
# [info] Added 3 missing timestamps to the 5 minute grid
# [info] Imputed 15 missing values in y
```

Values that are still missing after imputation can't be part of a training sample. By default fit raises:

```
** (ArgumentError) 44 training samples touch missing values that couldn't be imputed
(gaps longer than 30 steps). Set missing: %{drop_samples: true} to skip those samples,
or fill the gaps before fitting.
```

With `drop_samples: true` those samples are skipped, every other sample is trained on, and the log says how many were left out. A sample is skipped when its lag window or its forecast targets touch a missing value, in any column.

## Configuration

```elixir
Soothsayer.new(%{
  missing: %{
    impute: true,        # fill gaps at all (default true)
    impute_linear: 10,   # values filled linearly from each end of a gap
    impute_rolling: 10,  # extra window for the rolling mean pass
    drop_samples: false  # skip training samples touching what stays missing
  }
})
```

These are NeuralProphet's `impute_missing`, `impute_linear`, `impute_rolling` and `drop_missing`. With `impute: false` nothing is filled and only `drop_samples` decides between raising and skipping. Without auto-regression, `y` is never imputed (its rows are dropped), regressors still are.

## Regressors

Regressor and lagged regressor columns in the training data are imputed with the same limits, with or without auto-regression. Without auto-regression a regressor value that stays missing makes its row unusable, so the row is dropped with `drop_samples: true` and raises otherwise.

The regressors dataframe passed to `predict` is not imputed: a missing value there raises, naming the regressor and the timestamp, the same as a missing row.

## Predicting with history

The `history:` dataframe given to `predict` (observations newer than the training data, see [Auto-Regression](autoregression.md)) gets the same treatment as training data with lags: it is put on the frequency grid, rows at the end with a missing `y` are dropped, and the rest is imputed with the model's limits. The dropped tail is forecast like any other future step, from the last known value. What stays missing in the middle is unknown, and a timestamp whose lags reach into it gets NaN for `yhat` and `ar`. That includes the future: forecasting past the last known value needs its lags known, or every step after it is NaN.

The same holds for the training data itself: predicting the first `lags` timestamps of the training range, or the steps right after a gap that stayed open, gives NaN. Drop those rows before plotting.

## Backtest

`Soothsayer.backtest/3` fits on the first part of the data with everything above and forecasts the rest origin by origin. A validation date whose `y` is missing can't be scored, so it is left out of the metrics and of the predictions frame, but it still flows into the history the later forecasts are made from, where it is imputed. A forecast that comes out NaN, because its lags reach into a gap that stayed open, is left out the same way.

## Two decisions that differ from NeuralProphet

**Which samples `drop_samples` skips.** A sample is skipped when any column has a missing value anywhere in its widest window: the longest lag count among the target and the lagged regressors back from the origin, and `forecast_steps` ahead. NeuralProphet checks each input against its own window, so with a lagged regressor that has more lags than the AR component it can keep a sample whose target is missing only in that extra stretch. Soothsayer drops it. The difference is a handful of samples next to gaps longer than the imputation limits, and only in that configuration.

**Regressors at predict are not imputed.** NeuralProphet runs the same imputation over the regressors dataframe passed to `predict`. Soothsayer raises on a missing cell there, naming the regressor and the timestamp, the same as for a missing row. That frame is usually future values you assembled yourself, so a hole in it is more likely a bug upstream than a sensor outage, and a forecast quietly built on a filled-in regressor would be wrong without saying so. Fill it first with `Explorer.Series.fill_missing/2` if that is what you want.

## Log lines

Everything fit does to the data is logged at `:info`, and what it couldn't fill at `:warning`:

```
[info] Dropped 3 rows with missing y values
[info] Added 12 missing timestamps to the 1 hour grid
[info] Dropped 2 rows at the end with missing y values
[info] Imputed 40 missing values in y
[warning] 6 missing values remain in y after imputation, gaps longer than 30 steps aren't filled
[info] Skipped 51 training samples touching missing values
```

## A worked example

Hourly data with a short outage and a long one:

```elixir
hours = Enum.map(0..499, &NaiveDateTime.add(~N[2024-03-01 00:00:00], &1, :hour))

y =
  hours
  |> Enum.with_index()
  |> Enum.map(fn {_hour, index} ->
    cond do
      index in 100..104 -> nil          # five hours down
      index in 300..339 -> nil          # forty hours down
      true -> 20 + 5 * :math.sin(index / 24 * 2 * :math.pi())
    end
  end)

df = DataFrame.new(%{"ds" => hours, "y" => y})
model = Soothsayer.new(%{ar: %{enabled: true, lags: 24}, missing: %{drop_samples: true}})
fitted = Soothsayer.fit(model, df)
# [info] Imputed 25 missing values in y
# [warning] 20 missing values remain in y after imputation, gaps longer than 30 steps aren't filled
# [info] Skipped 44 training samples touching missing values
```

The five-hour gap is filled with a straight line. The forty-hour gap gets ten values from each side and its middle twenty stay open, so the 24 lag windows and the 1 step targets that reach into them are skipped: 20 missing hours plus 24 lags, as samples. Without `drop_samples: true` the same fit raises with that count.
