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

The honest test of an interval is coverage on data the model didn't see. `Soothsayer.backtest/3` reports it next to MAE and RMSE, from the outermost quantile columns:

```elixir
result = Soothsayer.backtest(model, df, horizon: 7)

result.metrics.coverage             # about 0.8 for a well calibrated 10/90 interval
result.metrics.mean_interval_width  # in the units of y
result.by_step[7].coverage          # a week ahead
```

`Soothsayer.cross_validate/3` gives the same numbers across several cutoffs through the history.

If coverage is much lower than the nominal level the intervals are too narrow: try more epochs, since the heads train alongside the median and may not have converged. If it's much higher they are too wide, which usually means the training data had noisier stretches than the holdout.

## Conformal prediction

Quantile heads learn the noise of the training data, and nothing forces a 10/90 band to cover 80% of what comes next. Conformal prediction fixes the width on data the model has not seen. Keep a calibration stretch right after the training data, let the fitted model forecast it, and the errors it makes there set the width:

```elixir
calibrated =
  model
  |> Soothsayer.fit(training)
  |> Soothsayer.calibrate(calibration, alpha: 0.1)

predictions = Soothsayer.predict(calibrated, future_dates)
predictions["yhat_lower"]
predictions["yhat_upper"]
```

`alpha` is the miss rate you accept, so `0.1` asks for a 90% interval. Under the usual conformal assumption, that the calibration rows and the future are exchangeable, the interval covers a new point with probability at least `1 - alpha`, whatever the model got wrong. Two methods:

- `method: :naive`, the default, scores each calibration row by `|y - yhat|` and builds `yhat -+ q_hat` around the point forecast. It needs no quantiles and gives a band of constant width.
- `method: :cqr`, conformalized quantile regression, needs `quantiles` on the model. It scores how far each row falls outside the band between the lowest and highest quantile, then pushes that band out by `q_hat`. The band keeps the shape the heads learned, wider where the series is noisier, and the calibration corrects its size. With `alpha: {0.05, 0.05}` each side gets its own score and its own correction.

`q_hat` is the `ceil((n + 1)(1 - alpha))`-th smallest score, so `alpha: 0.1` needs at least nine calibration rows and works better with a few hundred. The quantile columns stay as they were, the calibrated band lands in `yhat_lower` and `yhat_upper`. With auto-regression the calibration walks through the calibration frame the way `Soothsayer.backtest/3` does, one origin per row, and every step ahead gets its own `q_hat`, and rows forecast further out than `forecast_steps` were never calibrated and use the last step's. Pass `:events` and `:regressors` for the calibration dates the way you would to `predict/3`.

`Soothsayer.backtest/3` on a calibrated model reports the interval's `coverage` and `mean_interval_width` next to MAE and RMSE, and does the same from the outermost quantile columns when there is no calibration, which is the quick way to see whether the heads alone are honest.

Two details differ from NeuralProphet. It takes `scores[-int(n * alpha)]` as `q_hat`, which has no finite-sample correction and falls back to the smallest score when `n * alpha < 1`, and by default it overwrites the quantile columns in place.

## Limits

Quantile intervals reflect the noise the model saw during training. They don't widen for model misspecification, structural breaks, or for the compounding error of chained auto-regressive blocks past `forecast_steps`. Calibration corrects the size of the band but shares the last two blind spots, since the calibration stretch can only speak for what it contains.

## Next steps

- [Auto-regression](autoregression.md) for multi-step forecasting
- [The basics](basics.md) for the core concepts
