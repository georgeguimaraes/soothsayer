# Trends

The trend is the slow part of the series: growth or decline over months and years, with the seasonality and everything else riding on top of it. Soothsayer fits a piecewise linear trend, a straight line that can change slope at a fixed set of changepoints. A product launch, a pricing change or a new market usually shows up as one of those slope changes.

## How it works

```
trend(t) = k * t + m + sum(delta_j * f_j(t))
```

`k` is the base growth rate and `m` the offset, both learned. The `s_j` are the changepoint positions, picked from the data before training, and each `delta_j` is a learned slope adjustment attached to a hinge `f_j` that starts at `s_j`.

The shape of that hinge depends on `regularization`. Without it the hinge stops growing at the next changepoint, `f_j(t) = min(max(0, t - s_j), s_{j+1} - s_j)`, so `delta_j` is the slope of one segment relative to `k` and only that segment's data trains it. This is NeuralProphet's segmentwise trend, and it lets the slope bend sharply where the data does. With `regularization` set, the hinge is the cumulative Prophet one, `f_j(t) = max(0, t - s_j)`, where `delta_j` is the change of slope at `s_j`, which is the quantity an L1 penalty should shrink toward zero. `Soothsayer.Trend.basis/1` tells you which one a config uses.

The trend is the only component with an intercept. Seasonality, events and regressors are zero-centered offsets around it, so the trend carries the level of the series.

By default the trend is continuous: it bends at a changepoint but doesn't jump. With `growth: :discontinuous` it gets one more learned intercept per segment after the first, so the level can jump at a changepoint too. See [Discontinuous growth](#discontinuous-growth).

For the math behind the changepoints, see [NeuralProphet's trend docs](https://neuralprophet.com/html/trend.html).

## Configuration

```elixir
model = Soothsayer.new(%{
  trend: %{
    enabled: true,           # default: true
    changepoints: 10,        # potential changepoints, default: 10
    changepoints_range: 0.8, # place them in the first 80% of the data, default: 0.8
    growth: :linear,         # or :discontinuous to let the level jump at changepoints
    regularization: nil      # L1 penalty on slope changes, default: nil
  }
})
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Turn the trend component on or off |
| `changepoints` | `10` | Number of potential slope changes |
| `changepoints_range` | `0.8` | Fraction of the data where changepoints can sit |
| `growth` | `:linear` | `:linear` for a continuous trend, `:discontinuous` to allow jumps at changepoints |
| `regularization` | `nil` | L1 penalty on slope changes (and jumps), `nil` for none |

NeuralProphet's `growth: "off"` is `trend: %{enabled: false}` here: a flat line at the training mean.

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

## Discontinuous growth

Some series don't bend, they step: a price change, a new store, a tracking bug fixed. For those, let the trend jump at the changepoints:

```elixir
model = Soothsayer.new(%{
  trend: %{changepoints: 10, growth: :discontinuous}
})
```

This is NeuralProphet's discontinuous growth. Each segment after the first gets a learned intercept of its own, one extra trend input column per changepoint after the slope columns. Under the segmentwise basis (no regularization) those columns are one-hot segment indicators and the slope columns turn into ramps that only live inside their segment, so a segment's level and slope are trained by its own data and nobody else's. Under the cumulative basis (regularization on) they are steps that switch on at `s_j` and stay on, next to the cumulative hinges, so the L1 penalty means few jumps, the way it means few slope changes.

The jump is only allowed where a changepoint sits, so a step between two changepoints lands on the nearest one. Raise `changepoints` if the steps in your series are close together.

`Soothsayer.Trend.get_weights/1` returns the kernel with one row per input column: `t`, then the `changepoints` slope adjustments, then the `changepoints` intercepts. With `growth: :linear` the intercept rows aren't there.

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

Setting `regularization` also switches to the cumulative hinge described above, since the penalty only makes sense on slope changes. With `growth: :discontinuous` the penalty covers the jumps as well. Higher values mean fewer surviving changes and a smoother trend. There's no universal right value, so start around `0.1` and compare the trend column on a holdout against what you know about the series.

## Choosing parameters

The default 10 changepoints is a reasonable start. Raise it when you expect many slope changes, lower it or use `changepoints: 0` when the growth rate is steady.

Keep `changepoints_range` at 0.8 unless slope changes happen late in your data. Raising it lets the model react to a recent change, at the cost of extrapolating a slope fitted on fewer points.

Add `regularization` when the trend follows noise instead of the series. Start at `nil`, and if you set it, remember the delta semantics change as described above.

## Network architecture

With changepoints enabled, the trend input has shape `{batch_size, positions, 1 + changepoints}`, or `{batch_size, positions, 1 + 2 * changepoints}` with discontinuous growth, where `positions` is the number of timestamps in a training sample (one without auto-regression, `lags + forecast_steps` with it, see the [auto-regression guide](autoregression.md)). `Soothsayer.Trend.feature_count/1` gives the width for a config:

```elixir
# Per position:
# - Column 0: time t, scaled so the training data runs from 0 to 1
# - Columns 1-n: the changepoint hinges f_j(t), in the same units
# - Columns n+1-2n, discontinuous growth only: the segment intercepts, 0 or 1

input_shape = {nil, positions, Soothsayer.Trend.feature_count(config)}
```

The time columns are scaled by the training span rather than z-scored, as in NeuralProphet. Z-scoring each changepoint feature on its own would blow up the late ones (they are zero for most of the data) and let the slope of the last segment swing with the last few days, which is exactly the slope that gets extrapolated. The intercept columns are 0 or 1 and are left alone.

The [Livebook tutorial](https://github.com/georgeguimaraes/soothsayer/blob/main/livebook/soothsayer_tutorial.livemd) plots the network and the fitted trend.

## Next steps

- [Seasonality](seasonality.md) for yearly, weekly and daily patterns
- [Auto-regression](autoregression.md) for dependencies on recent values
