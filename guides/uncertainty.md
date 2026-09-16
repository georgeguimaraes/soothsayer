# Uncertainty

A point forecast says what you expect. A prediction interval says how wrong you might be. Soothsayer estimates intervals with quantile regression, the same approach NeuralProphet uses.

## Configuration

List the quantiles you want alongside the median:

```elixir
model = Soothsayer.new(%{
  quantiles: [0.1, 0.9]
})
```

`0.1` and `0.9` give an 80% interval: about one in ten observations should land below the lower line and one in ten above the upper. Use `[0.05, 0.95]` for 90%, `[0.25, 0.75]` for the interquartile range. Quantiles must be strictly between 0 and 1, and the median itself is always the `yhat` forecast.

## Reading the output

`Soothsayer.predict/3` returns one column per quantile next to the median, named after the quantile as a percentage:

```elixir
predictions = Soothsayer.predict(fitted, future_dates)

predictions["yhat"]     # median forecast
predictions["yhat_10"]  # lower line
predictions["yhat_90"]  # upper line
```

`0.025` becomes `yhat_2.5`, `0.975` becomes `yhat_97.5`. All in the units of `y`. `Soothsayer.predict_components/3` has the same numbers as tensors under `:quantiles`, a map from each configured quantile to its forecast, empty when none are configured.

## How it works

Every quantile gets its own linear head over the same inputs the components use: the trend features, the Fourier terms, events and regressors at every position of the sample (the lag timestamps and the forecast steps), plus the lags themselves. The head has one output per forecast step and learns how far that quantile sits from the median, so intervals can widen with the level of the series, with the horizon of a multi-step forecast, or around an event. With auto-regression, the regressor values at forecast steps you didn't ask for are the training mean, which the heads see too. That only affects the width of the interval, never the median.

Heads are trained with the pinball loss for their quantile:

```
loss = max(q * error, (q - 1) * error)     where error = y - prediction
```

For `q = 0.9` under-predicting costs nine times more than over-predicting, which pushes the head up until only 10% of points sit above it. The median keeps training on the Huber loss and is detached before the heads are added, so the quantile losses don't move it.

At prediction time upper quantiles are clipped to never fall below the median and lower quantiles to never rise above it, matching NeuralProphet's non-crossing rule.

## Checking calibration

The honest test of an interval is coverage on data the model didn't see:

```elixir
alias Explorer.Series

predictions = Soothsayer.predict(fitted, holdout_dates)
lower = Series.to_list(predictions["yhat_10"])
upper = Series.to_list(predictions["yhat_90"])

covered =
  [holdout_values, lower, upper]
  |> Enum.zip()
  |> Enum.count(fn {actual, low, high} -> actual >= low and actual <= high end)

covered / length(holdout_values)
# => about 0.8 for a well calibrated 10/90 interval
```

If coverage is much lower than the nominal level the intervals are too narrow: try more epochs, since the heads train alongside the median and may not have converged. If it's much higher they are too wide, which usually means the training data had noisier stretches than the holdout.

## Limits

Intervals reflect the noise the model saw during training. They don't widen for model misspecification, structural breaks, or for the compounding error of chained auto-regressive blocks past `forecast_steps`. NeuralProphet's conformal prediction, which calibrates intervals on a holdout set, is not implemented.

## Next steps

- [Auto-regression](autoregression.md) for multi-step forecasting
- [The basics](basics.md) for the core concepts
