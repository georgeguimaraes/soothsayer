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
    changepoints: 10,        # potential changepoints, default: 10, or a list of dates
    changepoints_range: 0.8, # place them in the first 80% of the data, default: 0.8
    growth: :linear,         # :discontinuous jumps at changepoints, :logistic saturates at a cap column
    regularization: nil      # L1 penalty on slope changes, default: nil
  }
})
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Turn the trend component on or off |
| `changepoints` | `10` | Number of potential slope changes, or a list of dates to bend at |
| `changepoints_range` | `0.8` | Fraction of the data where changepoints can sit |
| `growth` | `:linear` | `:linear` for a continuous trend, `:discontinuous` to allow jumps at changepoints, `:logistic` to saturate at a `cap` column |
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

Changepoints are spread evenly over the first `changepoints_range` of the training data the way NeuralProphet does it: `n + 1` points from the start, the first at zero, so the last changepoint sits at `changepoints_range * n / (n + 1)`, 73% of the way through with the defaults. Leaving the rest without one means the final slope is fitted on a decent stretch of data, and that final slope is what gets extrapolated into the forecast. Prophet puts its last changepoint at `changepoints_range` itself, and `changepoints_range: 0.88` gives you that tail here. Neither is better in general: across a dozen series the shorter tail won on Peyton Manning, retail sales and a daily series with a COVID break, the longer one on Wikipedia page views, sunspots and an hourly panel, each time by a wide margin. The tail segment is the slope your forecast extrapolates, so when the last stretch of your data has its own regime, raise `changepoints_range` or name the break.

## When you know where the break is

A grid is a guess. When you know the date a regime changed, a recession trough, a product launch, a pricing change, give the changepoints as dates instead of a count and the trend can bend exactly there and nowhere else:

```elixir
model = Soothsayer.new(%{
  trend: %{changepoints: [~D[2009-08-01]]}
})
```

The dates take the same type as your `ds` column, must be sorted and unique, and have to fall inside the training data. `changepoints_range` does nothing with a list. Combined with `growth: :discontinuous` a listed date can carry a level jump as well as a slope change, which is the honest model for a cliff like a lockdown.

This matters more than it sounds. On monthly US retail sales the default grid puts its last changepoint in January 2008, so the final segment has to describe the crash, the trough and the recovery with one line and the forecast undershoots by 4%. Two changepoints, one where the crash starts (January 2008) and one at the trough (August 2009), bring the error down to 1%. The first one matters too, since it keeps the pre-crash years from bending the line the recovery starts from. Any evenly spaced grid, here or in Prophet, is hostage to whether a point lands near the break. Named dates aren't.

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

`Soothsayer.Trend.get_weights/1` returns the kernel with one row per input column: `t`, then the `changepoints` slope adjustments, then the `changepoints` intercepts. With `growth: :linear` the intercept rows aren't there. A model with a local trend over [several series](series.md) returns one such map per series id.

## Saturating growth

Some series can't grow forever: market share, subscribers in a region, anything with a ceiling. `growth: :logistic` is Prophet's saturating trend: the piecewise linear trend the network learns becomes the exponent of a logistic curve, and the curve approaches a capacity you give per row in a `cap` column, with an optional `floor`:

```elixir
df = DataFrame.new(%{
  "ds" => dates,
  "y" => subscribers,
  "cap" => List.duplicate(50_000, length(dates))
})

model = Soothsayer.new(%{trend: %{growth: :logistic}})
fitted = Soothsayer.fit(model, df)

future = Soothsayer.future_timestamps(fitted, 90)
capacity = DataFrame.new(%{"ds" => future, "cap" => List.duplicate(50_000, 90)})
Soothsayer.predict(fitted, future, regressors: capacity)
```

The trend is `floor + (cap - floor) * sigmoid(trend)`, so changepoints bend how fast the series approaches the ceiling rather than the slope itself, and the capacity can change over time since it is a column, not a number. `cap` must be present at fit and, through `regressors:`, for every predicted date. A `floor` given at fit is required at predict too, and every cap has to sit above its floor. On an S-shaped series a linear trend keeps climbing past the ceiling where the logistic one levels off. Logistic growth has no discontinuous variant, and NeuralProphet does not offer it at all.

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

## Recent data first

Training weighs recent rows more than old ones. By default the last row of the training data counts twice as much in the loss as the oldest, with a smooth half-cosine ramp in between, so a slope that changed recently pulls the fit harder than the years before it. This is NeuralProphet's `newer_samples_weight`, on by default there too.

```elixir
model = Soothsayer.new(%{
  recency: %{weight: 5, start: 0.5}
})
```

`weight` is how many times more the last row counts than the rows before `start`, and `start` is the point in the training span, as a fraction, where the ramp begins: rows before it all share the lowest weight. `recency: %{enabled: false}` turns the weighting off and every row counts the same. The weights only touch the loss, the regularization penalties stay as they are, and the learning rate range test runs on the weighted loss as well.

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
