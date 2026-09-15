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
    regularization: nil, # L1 penalty on weights (default: nil)
    forecast_steps: 1    # Steps ahead forecast directly (default: 1)
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
| `forecast_steps` | `1` | How many steps ahead the AR head forecasts directly, see below |

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
- For multi-step forecasting, the model uses its own predictions as inputs, see below

## Forecasting Into the Future

Each AR prediction needs the `lags` values ending at its origin. For dates inside the training data the origin is the day before and the lags are real observations. For dates after the last observation, Soothsayer forecasts in blocks of `forecast_steps`: the first block directly from the last observation, the next block from the end of the first block using its predictions as lags, and so on up to the latest date you asked for.

```elixir
future_dates = Date.range(~D[2023-05-16], ~D[2023-06-14]) |> Enum.to_list()
predictions = Soothsayer.predict(fitted_with_ar, Series.from_list(future_dates))
```

With the default `forecast_steps: 1` every block is one day, so this is plain recursive forecasting: each day is predicted from the previous day's prediction and errors compound over the horizon. Trend, seasonality, events and regressors keep working at any horizon since they only depend on the date.

### Direct multi-step forecasting

Set `forecast_steps` to forecast several days directly from real observations instead of chaining predictions:

```elixir
model = Soothsayer.new(%{
  ar: %{enabled: true, lags: 14, forecast_steps: 7}
})
```

The AR output layer gets one unit per step, so the model learns a separate weight vector for "tomorrow", "two days out" and so on, all from the same lag window. Every training origin produces `forecast_steps` rows, one per step, and each row picks its unit with a one-hot step input. This is NeuralProphet's `n_forecasts`.

Within a block there is no error compounding: day 7 is predicted from the last 14 real values just like day 1, using the weights trained for a 7-day horizon. Past the block, the next block starts from predicted values, which is the manual unrolling NeuralProphet's maintainers recommend for going beyond `n_forecasts`.

Two practical notes. Direct forecasting for distant steps is harder, so accuracy on step 7 is lower than on step 1 whichever way you get there, and NeuralProphet suggests `forecast_steps` around half to one times `lags`. And each training origin needs `forecast_steps` days after it, so the last `forecast_steps - 1` days of your data only appear as targets, never as origins.

`Soothsayer.get_ar_weights/1` returns the output kernel as `{inputs, forecast_steps}`: row `i` is the i-th oldest lag, column `s` the weights for step `s + 1`.

### Forecasting from newer data

If you have observations newer than the training data, pass them with the `:history` option instead of refitting. History rows seed the lags and override training rows on the same dates:

```elixir
history = DataFrame.new(%{
  "ds" => Enum.to_list(Date.range(~D[2023-05-16], ~D[2023-06-14])),
  "y" => latest_observations
})

Soothsayer.predict(fitted_with_ar, Series.from_list([~D[2023-06-15]]), history: history)
```

### Assumptions

AR forecasting assumes gap-free data at the model's frequency (inferred at fit, see [The Basics](basics.md)). Lags are looked up one step back at a time, so a missing row inside the training data makes the rows right after it fall back to zero lags, and blocks always advance one step of the frequency: a day for daily data, an hour for hourly data.

Every requested timestamp has to sit on that grid. An hourly model can forecast 14:00 but not 14:30, and asking for the latter raises an error naming the timestamp and the frequency. Monthly data works on either month starts or month ends: a month-end timestamp always steps to the next or previous month end, so Jan 31, Feb 28 and Mar 31 form one grid.

When regressors are configured, the regressors dataframe must cover every timestamp in every block up to the latest requested one, since those get predicted too.

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
