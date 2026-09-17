# Several series

One model can learn from many related series at once: the sales of every store, the load of every meter, the traffic of every page. A series with little history borrows the shape of the others, and there is one fit to run instead of hundreds. This is NeuralProphet's global modeling with its `ID` column.

## Configuration

Put every series in one frame with a column that says which series each row belongs to, and name that column:

```elixir
model = Soothsayer.new(%{
  series: %{column: "id"}
})

df = DataFrame.new(%{
  "ds" => dates_a ++ dates_b,
  "y" => sales_a ++ sales_b,
  "id" => List.duplicate("store_a", 365) ++ List.duplicate("store_b", 365)
})

fitted = Soothsayer.fit(model, df)
```

Ids are strings. Every series needs at least two rows, is sorted on its own, and has its gaps handled on its own, see [Missing data](missing_data.md). All series must have the same frequency.

## Predicting

Prediction takes a frame with `ds` and the id column instead of a series of dates, and the output carries the id column right after `ds`. Rows can come in any order and any mix of series:

```elixir
future = DataFrame.new(%{
  "ds" => [~D[2024-01-01], ~D[2024-01-01], ~D[2024-01-02]],
  "id" => ["store_a", "store_b", "store_a"]
})

Soothsayer.predict(fitted, future)
# => ds, id, yhat, trend, yearly_seasonality, ...
```

An id the model was not fitted on raises. The `:history` and `:regressors` frames carry the id column too, so each series gets its own recent observations and its own regressor values. `Soothsayer.backtest/3` holds out the tail of every series and walks each one separately; its predictions frame has the id column first.

## What is shared and what isn't

One network is trained on the samples of every series. By default the trend, the seasonalities, the events, the regressors and the auto-regression are shared: the same coefficients apply to every series. What each series keeps for itself:

- its scale. With `normalize: :local`, the default and NeuralProphet's, `y` is z-scored per series, so a store selling ten times more than another lands on the same footing and the shared components describe the shape both follow. `normalize: :global` scales every series with one mean and standard deviation, which only makes sense when the series really live on the same scale.
- its own past for the lags. Auto-regression reads each series' observations, never a neighbour's.
- its regressor and condition values.

The time axis is shared: numeric time starts at the earliest timestamp of any series and the changepoints are placed over the union of all timestamps, so a series that starts late sees the same changepoints as the others. The `:auto` seasonality decisions and the country holidays are settled on that union as well. Events are shared too: a date in the events frame is an event for every series.

The quantile heads see which series a sample belongs to, so prediction intervals can be wider for a noisier series.

## Local trend and seasonality

Series that grow at their own pace or peak in different months can't share a trend or a seasonality. `trend: :local` gives every series its own trend kernel and intercept, `seasonality: :local` its own Fourier coefficients for every period, while everything else stays shared:

```elixir
model = Soothsayer.new(%{
  series: %{
    column: "id",
    trend: :local,
    seasonality: :local,
    local_regularization: 0.1
  }
})
```

Under the hood the kernel of a local layer has one slice per series, and each sample multiplies its features by the slice its one-hot picks, so the weights still train together in one pass. `local_regularization` is NeuralProphet's "glocal" mode: a penalty of `lambda * mean((kernel - mean kernel across series)^2)` pulls every series' kernel toward the average, so a series with little data leans on the others and a series with plenty can still differ. Start small, around `0.1`, since a large value flattens the differences you asked the local mode for. The intercepts are left out of the penalty, so levels stay apart when the series are scaled together.

A local trend is also what lets `normalize: :global` work: with one scale for all series the shared trend can only describe one level, and a local intercept carries each series' own.

`Soothsayer.Trend.get_weights/1` on a local trend returns a map from id to that series' `kernel` and `bias`, and `params.data["yearly_dense"]["kernel"]` and friends have the series axis first.

## Reading the effects

`Soothsayer.get_event_effects/1`, `get_regressor_effects/1` and `get_ar_weights/1` return the shared coefficients as for a single series. Each series' scale is on the model under its training data, so a coefficient in normalized units means something different in the units of each series.

## Compared with NeuralProphet

NeuralProphet lets the events frame carry an `ID` column for per-series events and can forecast an unknown id with the global normalization when asked to. Both are left out: events are shared and an unknown id raises. Time normalization is always global here, matching NeuralProphet's default. Local mode applies to the whole trend and to every seasonal period at once, where NeuralProphet has a switch per period, and the trend intercept is per series in local mode where NeuralProphet keeps one. The penalty is the same squared distance to the mean kernel, applied from the first step.

## Next steps

- [The basics](basics.md) for fitting and predicting one series
- [Auto-regression](autoregression.md) for how the lags are seeded
- [Uncertainty](uncertainty.md) for quantiles and calibration, which pool the calibration rows of every series
