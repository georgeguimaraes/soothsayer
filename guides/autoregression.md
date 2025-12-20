# Auto-Regression

Auto-regression (AR) captures dependencies on recent values. Enable this when today's value depends on yesterday's (or the last few days).

This is useful for:
- Financial data with momentum
- Sensor readings with inertia
- Sales with carry-over effects
- Any data where values persist

## How It Works

The AR component models the current value as depending on previous values:

```
ar(t) = sum(w_i * y(t-i))
```

Where:
- `y(t-i)` = value at lag i (1 to lags)
- `w_i` = learned weight for each lag

For more details, see [NeuralProphet's Auto-Regression documentation](https://neuralprophet.com/html/autoregression.html).

## Configuration

```elixir
model = Soothsayer.new(%{
  ar: %{
    enabled: true,       # Enable AR component (default: false)
    lags: 7,           # Number of lagged values to use
    layers: [],          # Hidden layers for deep AR-Net (default: [])
    regularization: nil  # L1 penalty on weights (default: nil)
  }
})
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enable/disable AR component |
| `lags` | `0` | Number of lagged values to use |
| `layers` | `[]` | Hidden layer sizes for deep AR-Net |
| `regularization` | `nil` | L1 penalty to encourage sparsity |

## Linear AR

For simple linear dependencies:

```elixir
model = Soothsayer.new(%{
  ar: %{enabled: true, lags: 7}
})
```

This learns a weighted sum of the last 7 values.

## Deep AR-Net

For non-linear relationships, add hidden layers:

```elixir
model = Soothsayer.new(%{
  ar: %{
    enabled: true,
    lags: 7,
    layers: [32, 16]  # Two hidden layers with ReLU activation
  }
})
```

Use deep AR-Net when:
- Linear AR doesn't capture the pattern
- You have enough data to train a deeper model
- The relationship between past and future is complex

## Choosing lags

Start with the natural cycle of your data:

| Data Frequency | Suggested lags | Reason |
|----------------|------------------|--------|
| Daily with weekly pattern | 7 | Captures full week |
| Daily with monthly pattern | 30 | Captures full month |
| Hourly with daily pattern | 24 | Captures full day |

You can also examine autocorrelation to see which lags are useful.

## Example: AR(1) Process

```elixir
alias Explorer.DataFrame
alias Explorer.Series

# Generate AR(1) data: each value depends on the previous
:rand.seed(:exsss, {42, 42, 42})

n_days = 500
dates = Enum.map(0..(n_days - 1), fn i -> Date.add(~D[2022-01-01], i) end)

y = Enum.reduce(1..(n_days - 1), [100.0], fn _i, [prev | _] = acc ->
  trend = 0.1
  ar = 0.7 * (prev - 100)  # Mean-reverting AR(1)
  noise = :rand.normal(0, 5)
  [100 + trend * length(acc) + ar + noise | acc]
end)
|> Enum.reverse()

df = DataFrame.new(%{"ds" => dates, "y" => y})

# Model WITHOUT AR (trend only)
model_no_ar = Soothsayer.new(%{
  trend: %{changepoints: 5},
  seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
  ar: %{enabled: false},
  epochs: 30
})

# Model WITH AR
model_with_ar = Soothsayer.new(%{
  trend: %{changepoints: 5},
  seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
  ar: %{enabled: true, lags: 7},
  epochs: 30
})

fitted_no_ar = Soothsayer.fit(model_no_ar, df)
fitted_with_ar = Soothsayer.fit(model_with_ar, df)
```

The AR model will track short-term fluctuations much better.

## Inspecting AR Weights

Use `Soothsayer.get_ar_weights/1` to see which lags are important:

```elixir
ar_weights = Soothsayer.get_ar_weights(fitted_with_ar)

# For linear AR
kernel = ar_weights["ar_dense_out"].kernel
# => #Nx.Tensor<f32[7][1]>

# View weights
weights = Nx.to_flat_list(kernel)
# => [-0.12, 0.45, 0.08, ...] # Weight for each lag
```

Higher absolute weight = more important lag.

### Deep AR-Net Weights

For models with hidden layers:

```elixir
model = Soothsayer.new(%{
  ar: %{enabled: true, lags: 7, layers: [32, 16]}
})
fitted = Soothsayer.fit(model, df)

weights = Soothsayer.get_ar_weights(fitted)
# => %{
#   "ar_dense_0" => %{kernel: ..., bias: ...},  # First hidden layer
#   "ar_dense_1" => %{kernel: ..., bias: ...},  # Second hidden layer
#   "ar_dense_out" => %{kernel: ..., bias: ...} # Output layer
# }
```

## Regularization

When you're unsure how many lags matter, use regularization:

```elixir
model = Soothsayer.new(%{
  ar: %{
    enabled: true,
    lags: 14,          # More lags than we likely need
    regularization: 0.1  # L1 penalty
  }
})
```

Regularization pushes unimportant lag weights toward zero, effectively selecting which lags matter.

| Regularization | Effect |
|----------------|--------|
| `nil` or `0` | No penalty, all lags can have any weight |
| `0.01 - 0.1` | Light penalty, minor lags zeroed |
| `0.1 - 1.0` | Strong penalty, only dominant lags remain |

## Data Considerations

**Training Data:**
- First `lags` observations are used to seed the AR model
- Training targets start at observation `lags + 1`
- More lags = less effective training data

**Prediction:**
- Predictions use observed values from training data as context
- For multi-step forecasting, the model uses its own predictions as inputs

## Network Architecture

The AR component adds an input branch to the network:

```elixir
# Input shape
ar_input_shape = {nil, lags}

# For linear AR: direct dense layer to output
# For deep AR-Net: hidden layers -> dense output
```

## Next Steps

- [Events](events.md) - Holidays and special occasions
- [Trends](trends.md) - Piecewise linear trends with changepoints
- [Seasonality](seasonality.md) - Yearly and weekly patterns
- [The Basics](basics.md) - Fundamental concepts
