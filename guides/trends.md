# Trends

The trend component captures long-term growth or decline in your data. Soothsayer uses piecewise linear trends with automatic changepoint detection, allowing the trend to change slope at multiple points.

This is useful for capturing:
- Product launches that accelerate growth
- Market shifts that slow growth
- Policy changes that affect trajectory
- Seasonal baseline shifts

## How It Works

The trend is modeled as:

```
trend(t) = k * t + m + sum(delta_j * max(0, t - s_j))
```

Where:
- `k` = base growth rate (learned)
- `m` = offset (learned)
- `s_j` = changepoint positions (computed from data)
- `delta_j` = rate adjustments at each changepoint (learned)

The model learns which changepoints matter and how much the slope changes at each one.

For more details on the math, see [NeuralProphet's Trend documentation](https://neuralprophet.com/html/trend.html).

## Configuration

```elixir
model = Soothsayer.new(%{
  trend: %{
    enabled: true,           # Enable trend component (default: true)
    n_changepoints: 10,      # Number of potential changepoints (default: 10)
    changepoints_range: 0.8, # Place in first 80% of data (default: 0.8)
    regularization: nil      # L1 penalty on rate changes (default: nil)
  }
})
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable the trend component |
| `n_changepoints` | `10` | Number of potential slope changes |
| `changepoints_range` | `0.8` | Fraction of data where changepoints can occur |
| `regularization` | `nil` | L1 penalty to encourage sparse changepoints |

## Simple Linear Trend

For data with a consistent growth rate, disable changepoints:

```elixir
model = Soothsayer.new(%{
  trend: %{n_changepoints: 0}
})
```

This gives you a simple linear trend: `trend(t) = k * t + m`

## Piecewise Linear Trend

For data where the growth rate changes over time:

```elixir
model = Soothsayer.new(%{
  trend: %{n_changepoints: 10, changepoints_range: 0.8}
})
```

The `changepoints_range` parameter controls where changepoints can be placed. Setting it to `0.8` means changepoints are only placed in the first 80% of the data, preventing overfitting at the end of the series.

## Example: Detecting Slope Changes

```elixir
alias Explorer.DataFrame

# Data with a slope change: flat first year, steep second year
n_days = 730
dates = Enum.map(0..(n_days - 1), fn i -> Date.add(~D[2020-01-01], i) end)

y = Enum.map(0..(n_days - 1), fn i ->
  # Slope changes from 0.1 to 3.0 at day 365
  trend = if i < 365, do: 100 + 0.1 * i, else: 100 + 0.1 * 365 + 3.0 * (i - 365)
  noise = :rand.normal(0, 10)
  trend + noise
end)

df = DataFrame.new(%{"ds" => dates, "y" => y})

# Model WITHOUT changepoints (misses the slope change)
model_linear = Soothsayer.new(%{
  trend: %{n_changepoints: 0},
  seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
  epochs: 100
})

# Model WITH changepoints (captures the slope change)
model_piecewise = Soothsayer.new(%{
  trend: %{n_changepoints: 10, changepoints_range: 0.8},
  seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
  epochs: 100
})

fitted_linear = Soothsayer.fit(model_linear, df)
fitted_piecewise = Soothsayer.fit(model_piecewise, df)
```

The piecewise model will capture the slope change much better than the simple linear model.

## Regularization

When you're unsure how many changepoints you need, set more than necessary and use regularization to prune unimportant ones:

```elixir
model = Soothsayer.new(%{
  trend: %{
    n_changepoints: 25,      # More than we likely need
    regularization: 0.1      # L1 penalty pushes small changes toward zero
  }
})
```

Higher regularization values encourage sparser changepoints (fewer slope changes).

| Regularization | Effect |
|----------------|--------|
| `nil` or `0` | No penalty, all changepoints can have any value |
| `0.01 - 0.1` | Light penalty, subtle changes may be zeroed out |
| `0.1 - 1.0` | Strong penalty, only significant changes remain |

## Choosing Parameters

**n_changepoints:**
- Start with the default (10)
- Increase if you expect many slope changes
- Decrease if you expect a smooth trend
- Use `n_changepoints: 0` for simple linear trend

**changepoints_range:**
- Default (0.8) works well for most cases
- Decrease if you have a short forecast horizon
- Increase if slope changes occur late in your data

**regularization:**
- Start with `nil` (no regularization)
- Add if you see overfitting (trend follows noise too closely)
- Higher values = smoother trend with fewer changes

## Network Architecture

With changepoints enabled, the trend input has shape `{batch_size, 1 + n_changepoints}`:

```elixir
# The network receives:
# - Column 0: normalized time t
# - Columns 1-n: changepoint features max(0, t - s_j)

input_shape = {nil, 1 + n_changepoints}
```

See the [Interactive Livebook Tutorial](https://github.com/georgeguimaraes/soothsayer/blob/main/livebook/soothsayer_tutorial.livemd) for network visualization examples.

## Next Steps

- [Seasonality](seasonality.md) - Add yearly and weekly patterns
- [Auto-Regression](autoregression.md) - Capture dependencies on recent values
