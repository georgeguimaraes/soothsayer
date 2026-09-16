# Trends

The trend is the slow part of the series: growth or decline over months and years, with the seasonality and everything else riding on top of it. Soothsayer fits a piecewise linear trend, a straight line that can change slope at a fixed set of changepoints. A product launch, a pricing change or a new market usually shows up as one of those slope changes.

## How it works

```
trend(t) = k * t + m + sum(delta_j * f_j(t))
```

`k` is the base growth rate and `m` the offset, both learned. The `s_j` are the changepoint positions, picked from the data before training, and each `delta_j` is a learned slope adjustment attached to a hinge `f_j` that starts at `s_j`.

The shape of that hinge depends on `regularization`. Without it the hinge stops growing at the next changepoint, `f_j(t) = min(max(0, t - s_j), s_{j+1} - s_j)`, so `delta_j` is the slope of one segment relative to `k` and only that segment's data trains it. This is NeuralProphet's segmentwise trend, and it lets the slope bend sharply where the data does. With `regularization` set, the hinge is the cumulative Prophet one, `f_j(t) = max(0, t - s_j)`, where `delta_j` is the change of slope at `s_j`, which is the quantity an L1 penalty should shrink toward zero. `Soothsayer.Trend.basis/1` tells you which one a config uses.

The trend is the only component with an intercept. Seasonality, events and regressors are zero-centered offsets around it, so the trend carries the level of the series.

For the math behind the changepoints, see [NeuralProphet's trend docs](https://neuralprophet.com/html/trend.html).

## Configuration

```elixir
model = Soothsayer.new(%{
  trend: %{
    enabled: true,           # default: true
    changepoints: 10,        # potential changepoints, default: 10
    changepoints_range: 0.8, # place them in the first 80% of the data, default: 0.8
    regularization: nil      # L1 penalty on slope changes, default: nil
  }
})
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Turn the trend component on or off |
| `changepoints` | `10` | Number of potential slope changes |
| `changepoints_range` | `0.8` | Fraction of the data where changepoints can sit |
| `regularization` | `nil` | L1 penalty on slope changes, `nil` for none |

## Linear trend

For a series with one steady growth rate, drop the changepoints:

```elixir
model = Soothsayer.new(%{
  trend: %{changepoints: 0}
})
```

That leaves `trend(t) = k * t + m`.

## Piecewise linear trend

For a series whose growth rate changes over time:

```elixir
model = Soothsayer.new(%{
  trend: %{changepoints: 10, changepoints_range: 0.8}
})
```

Changepoints are spread evenly over the first `changepoints_range` of the training data. Leaving the last 20% without one means the final slope is fitted on a decent stretch of data, and that final slope is what gets extrapolated into the forecast.

## Example: a slope change

```elixir
alias Explorer.DataFrame

# Flat first year, steep second year
n_days = 730
dates = Enum.map(0..(n_days - 1), fn i -> Date.add(~D[2020-01-01], i) end)

y = Enum.map(0..(n_days - 1), fn i ->
  # Slope goes from 0.1 to 3.0 at day 365
  trend = if i < 365, do: 100 + 0.1 * i, else: 100 + 0.1 * 365 + 3.0 * (i - 365)
  noise = :rand.normal(0, 10)
  trend + noise
end)

df = DataFrame.new(%{"ds" => dates, "y" => y})

# Without changepoints, one line through both years
model_linear = Soothsayer.new(%{
  trend: %{changepoints: 0},
  seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
  epochs: 100
})

# With changepoints, the bend at day 365 is fitted
model_piecewise = Soothsayer.new(%{
  trend: %{changepoints: 10, changepoints_range: 0.8},
  seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
  epochs: 100
})

fitted_linear = Soothsayer.fit(model_linear, df)
fitted_piecewise = Soothsayer.fit(model_piecewise, df)
```

The trend column of `Soothsayer.predict/3` shows the difference: the linear model averages the two slopes, the piecewise one follows the bend.

## Regularization

When you don't know how many slope changes to expect, set more changepoints than you need and let an L1 penalty zero out the ones that don't earn their keep:

```elixir
model = Soothsayer.new(%{
  trend: %{
    changepoints: 25,
    regularization: 0.1
  }
})
```

Setting `regularization` also switches to the cumulative hinge described above, since the penalty only makes sense on slope changes. Higher values mean fewer surviving changes and a smoother trend. There's no universal right value, so start around `0.1` and compare the trend column on a holdout against what you know about the series.

## Choosing parameters

The default 10 changepoints is a reasonable start. Raise it when you expect many slope changes, lower it or use `changepoints: 0` when the growth rate is steady.

Keep `changepoints_range` at 0.8 unless slope changes happen late in your data. Raising it lets the model react to a recent change, at the cost of extrapolating a slope fitted on fewer points.

Add `regularization` when the trend follows noise instead of the series. Start at `nil`, and if you set it, remember the delta semantics change as described above.

## Network architecture

With changepoints enabled, the trend input has shape `{batch_size, positions, 1 + changepoints}`, where `positions` is the number of timestamps in a training sample (one without auto-regression, `lags + forecast_steps` with it, see the [auto-regression guide](autoregression.md)):

```elixir
# Per position:
# - Column 0: time t, scaled so the training data runs from 0 to 1
# - Columns 1-n: the changepoint hinges f_j(t), in the same units

input_shape = {nil, positions, 1 + changepoints}
```

The time features are scaled by the training span rather than z-scored, as in NeuralProphet. Z-scoring each changepoint feature on its own would blow up the late ones (they are zero for most of the data) and let the slope of the last segment swing with the last few days, which is exactly the slope that gets extrapolated.

The [Livebook tutorial](https://github.com/georgeguimaraes/soothsayer/blob/main/livebook/soothsayer_tutorial.livemd) plots the network and the fitted trend.

## Next steps

- [Seasonality](seasonality.md) for yearly, weekly and daily patterns
- [Auto-regression](autoregression.md) for dependencies on recent values
