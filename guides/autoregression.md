# Auto-regression

Auto-regression (AR) uses the recent values of the series itself. Turn it on when today's value depends on yesterday's, or on the last few days: sales with carry-over, sensor readings with inertia, anything where a high value tends to be followed by another high value.

## How it works

The AR component models what the other components leave over. Each lag is stationarized first, by subtracting the trend, seasonality, events and regressors evaluated at that lag's own timestamp, and the AR network works on those residuals:

```
residual(t-i) = y(t-i) - trend(t-i) - seasonality(t-i) - events(t-i) - regressors(t-i)
ar(t) = sum(w_i * residual(t-i))
```

Where `y(t-i)` is the value at lag `i` (1 to `lags`) and `w_i` the learned weight for that lag.

This is what NeuralProphet does too, and it matters more than it looks. If the AR network saw the raw lags it would happily absorb the level and the daily cycle as well, the trend would go flat at the training mean, and multi-step forecasts would drift back to that mean. With residual lags the trend has to carry the level and the seasonalities their cycles, so the components stay interpretable and a forecast a few steps out follows where the series actually is.

For the math, see [NeuralProphet's auto-regression documentation](https://neuralprophet.com/html/autoregression.html).

## Configuration

```elixir
model = Soothsayer.new(%{
  ar: %{
    enabled: true,       # default: false
    lags: 7,             # how many past values to use
    layers: [],          # hidden layers for a deep AR-Net (default: [])
    regularization: nil, # L1 penalty on the weights (default: nil)
    forecast_steps: 1    # steps forecast directly from one lag window (default: 1)
  }
})
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Turn the AR component on |
| `lags` | `0` | Number of past values fed to the network |
| `layers` | `[]` | Hidden layer sizes for a deep AR-Net |
| `regularization` | `nil` | L1 penalty that pushes unused lag weights to zero |
| `forecast_steps` | `1` | How many steps ahead the AR head forecasts directly, see below |

## Linear AR

With no hidden layers the component is a weighted sum of the last `lags` values:

```elixir
model = Soothsayer.new(%{
  ar: %{enabled: true, lags: 7}
})
```

## Deep AR-Net

Hidden layers with ReLU activation let the component learn non-linear relationships between past and future:

```elixir
model = Soothsayer.new(%{
  ar: %{
    enabled: true,
    lags: 7,
    layers: [32, 16]
  }
})
```

Reach for it when linear AR leaves a pattern on the table and you have enough data to train the extra weights.

## Choosing lags

Start with the natural cycle of your data: 7 for daily data with a weekly pattern, 30 for a monthly one, 24 for hourly data with a daily pattern. An autocorrelation plot shows which lags carry information if you want to be more precise.

## Example: an AR(1) process

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

The AR model tracks the short-term swings, the trend-only model can only draw a line through them.

## Inspecting AR weights

`Soothsayer.get_ar_weights/1` returns the weights of each layer:

```elixir
ar_weights = Soothsayer.get_ar_weights(fitted_with_ar)

# For linear AR
kernel = ar_weights["ar_dense_out"].kernel
# => #Nx.Tensor<f32[7][1]>

weights = Nx.to_flat_list(kernel)
# => [-0.12, 0.45, 0.08, ...] # one weight per lag, oldest first
```

The bigger a weight's absolute value, the more that lag drives the forecast.

For a deep AR-Net there is one entry per layer:

```elixir
model = Soothsayer.new(%{
  ar: %{enabled: true, lags: 7, layers: [32, 16]}
})
fitted = Soothsayer.fit(model, df)

weights = Soothsayer.get_ar_weights(fitted)
# => %{
#   "ar_dense_0" => %{kernel: ..., bias: ...},  # First hidden layer
#   "ar_dense_1" => %{kernel: ..., bias: ...},  # Second hidden layer
#   "ar_dense_out" => %{kernel: ...}            # Output layer, no bias
# }
```

## Regularization

When you don't know how many lags matter, give the model more than it needs and let the L1 penalty sort them out:

```elixir
model = Soothsayer.new(%{
  ar: %{
    enabled: true,
    lags: 14,
    regularization: 0.1
  }
})
```

The penalty pushes the weights of lags that don't help toward zero, so reading the weights afterwards tells you which lags the data supports. `nil` or `0` means no penalty. How strong a value has to be depends on the scale of the residuals, so treat the `0.1` above as a starting point and look at the weights.

## What the training data has to cover

The first `lags` observations only seed the lag windows, so training targets start at observation `lags + 1`. More lags means fewer training samples from the same data. At predict, the lags come from the training data for dates inside it, and from the model's own predictions for dates past the end, see below.

## Forecasting into the future

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

The AR output layer gets one unit per step, so the model learns a separate weight vector for "tomorrow", "two days out" and so on, all from the same lag window. Every training origin is one sample whose `forecast_steps` targets share that window, and the loss averages over all of them. This is NeuralProphet's `n_forecasts`.

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

AR works on the model's frequency grid (inferred at fit, see [The Basics](basics.md)): lags are looked up one step back at a time and blocks always advance one step, a day for daily data, an hour for hourly data. Fit puts the training data on that grid and imputes the gaps, see [Missing Data](missing_data.md), so the lags it learns from are consecutive steps. A value that stays missing after imputation (a gap longer than the imputation limits, in training data or in `history:`) is unknown at predict, and a timestamp whose lag window reaches into it gets NaN for `yhat` and `ar` rather than a forecast built on made-up lags. The first `lags` timestamps of the training data are NaN for the same reason, so when you predict over the whole training range to plot it, drop those rows first. Forecasting past the last observation needs its `lags` values known too, or every step after it is NaN.

Every requested timestamp has to sit on that grid. An hourly model can forecast 14:00 but not 14:30, and asking for the latter raises an error naming the timestamp and the frequency. Monthly data works on either month starts or month ends: a month-end timestamp always steps to the next or previous month end, so Jan 31, Feb 28 and Mar 31 form one grid.

When regressors are configured, the regressors dataframe must cover every timestamp in every block up to the latest requested one, since those get predicted too. It doesn't have to cover the rest of the last block past that timestamp: a regressor value only affects its own timestamp, so the steps nobody asked for are filled with the training mean. Values from the training period are remembered by the model, which is how the lags of an early forecast reach back into it.

## Network architecture

Every sample the network sees is one forecast origin laid out as `lags + forecast_steps` positions: the lag timestamps, oldest first, then the target timestamps. The time-based components (trend, seasonality, events, regressors) take `{batch, positions, features}` inputs and produce one value per position through a single shared linear layer. The AR branch takes the raw lags, `{batch, lags}`, subtracts the other components at the lag positions, and outputs `{batch, forecast_steps}`:

```elixir
# Input shapes
ar_input_shape = {nil, lags}
trend_input_shape = {nil, lags + forecast_steps, 1 + changepoints}

# AR branch
# lags - (trend + seasonality + events + regressors at the lag positions)
# -> hidden layers (deep AR-Net only) -> dense with forecast_steps units
```

The component outputs are their values at the target positions, so they still add up to the combined forecast.

## Related guides

- [Events](events.md) for holidays and one-off dates
- [Trends](trends.md) for piecewise linear trends with changepoints
- [Seasonality](seasonality.md) for yearly and weekly patterns
- [Missing Data](missing_data.md) for how gaps are filled before the lags are built
