# The Basics

This tutorial covers the fundamentals of time series forecasting with Soothsayer. You'll learn how to create a model, fit it to data, and make predictions.

## Data Format

Soothsayer expects an Explorer DataFrame with two columns:

- `ds` - dates (Date type)
- `y` - target values (numeric)

```elixir
alias Explorer.DataFrame

df = DataFrame.new(%{
  "ds" => [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]],
  "y" => [100.0, 102.5, 101.3]
})
```

## Creating a Model

Use `Soothsayer.new/1` to create a model with your configuration:

```elixir
model = Soothsayer.new(%{
  trend: %{enabled: true},
  seasonality: %{
    yearly: %{enabled: true, fourier_terms: 6},
    weekly: %{enabled: true, fourier_terms: 3}
  },
  epochs: 100,
  learning_rate: 0.01
})
```

### Default Configuration

If you call `Soothsayer.new()` without arguments, you get sensible defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trend.enabled` | `true` | Enable trend component |
| `trend.changepoints` | `10` | Number of potential changepoints |
| `trend.changepoints_range` | `0.8` | Place changepoints in first 80% of data |
| `seasonality.yearly.enabled` | `true` | Enable yearly seasonality |
| `seasonality.yearly.fourier_terms` | `6` | Flexibility of yearly pattern |
| `seasonality.weekly.enabled` | `true` | Enable weekly seasonality |
| `seasonality.weekly.fourier_terms` | `3` | Flexibility of weekly pattern |
| `epochs` | `100` | Training iterations |
| `learning_rate` | `0.01` | How fast to learn |

## Fitting the Model

Use `Soothsayer.fit/2` to train the model on your data:

```elixir
fitted_model = Soothsayer.fit(model, df)
```

Training uses [EXLA](https://hexdocs.pm/exla) by default for fast execution on CPU/GPU. Make sure EXLA is configured:

```elixir
# In config/config.exs
config :nx, default_backend: EXLA.Backend

# Or at runtime
Nx.global_default_backend(EXLA.Backend)
```

### GPU Memory Configuration

By default, XLA pre-allocates 90% of GPU memory at startup. If you're sharing the GPU with other applications (X windows, other ML processes, etc.) and see `CUDNN_STATUS_INTERNAL_ERROR` or out-of-memory errors, disable preallocation:

```elixir
# In config/config.exs
config :exla, :clients,
  cuda: [platform: :cuda, preallocate: false]
```

Or via environment variable (before starting):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

See the [README](https://github.com/georgeguimaraes/soothsayer#gpu-memory-configuration) for more options.

## Making Predictions

Use `Soothsayer.predict/2` with an Explorer Series of dates:

```elixir
alias Explorer.Series

future_dates = Series.from_list([~D[2023-01-04], ~D[2023-01-05], ~D[2023-01-06]])
predictions = Soothsayer.predict(fitted_model, future_dates)
# => #Nx.Tensor<f32[3][1]>
```

The result is an Nx tensor with shape `{n_dates, 1}`.

## Getting Components

One of Soothsayer's strengths is interpretability. Use `Soothsayer.predict_components/2` to see what each component contributes:

```elixir
components = Soothsayer.predict_components(fitted_model, future_dates)
# => %{
#   combined: #Nx.Tensor<...>,
#   trend: #Nx.Tensor<...>,
#   yearly_seasonality: #Nx.Tensor<...>,
#   weekly_seasonality: #Nx.Tensor<...>,
#   ar: #Nx.Tensor<...>
# }
```

This helps you understand:
- Is the forecast driven by trend or seasonality?
- How much does each seasonal pattern contribute?
- What's the impact of auto-regression?

## Complete Example

```elixir
alias Explorer.DataFrame
alias Explorer.Series

# Generate synthetic data with trend and seasonality
:rand.seed(:exsss, {42, 42, 42})

start_date = ~D[2020-01-01]
dates = Date.range(start_date, ~D[2022-12-31])

y = Enum.map(dates, fn date ->
  days = Date.diff(date, start_date)
  trend = 1000 + 0.5 * days
  yearly = 50 * :math.sin(2 * :math.pi() * days / 365.25)
  weekly = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
  noise = :rand.normal(0, 30)
  trend + yearly + weekly + noise
end)

df = DataFrame.new(%{"ds" => dates, "y" => y})

# Create and fit model
model = Soothsayer.new(%{epochs: 50})
fitted_model = Soothsayer.fit(model, df)

# Predict on training data
predictions = Soothsayer.predict(fitted_model, df["ds"])

# Get components
components = Soothsayer.predict_components(fitted_model, df["ds"])
```

## Visualizing Results

If you're using Livebook, you can visualize with VegaLite:

```elixir
alias VegaLite, as: Vl

df_with_predictions = DataFrame.put(df, "yhat", predictions)

Vl.new(width: 800, height: 400, title: "Actual vs Predicted")
|> Vl.data_from_values(df_with_predictions, only: ["ds", "y", "yhat"])
|> Vl.layers([
  Vl.new()
  |> Vl.mark(:point, opacity: 0.3)
  |> Vl.encode_field(:x, "ds", type: :temporal)
  |> Vl.encode_field(:y, "y", type: :quantitative),
  Vl.new()
  |> Vl.mark(:line, color: "tomato", stroke_width: 2)
  |> Vl.encode_field(:x, "ds", type: :temporal)
  |> Vl.encode_field(:y, "yhat", type: :quantitative)
])
```

See the [Interactive Livebook Tutorial](https://github.com/georgeguimaraes/soothsayer/blob/main/livebook/soothsayer_tutorial.livemd) for more visualization examples.

## Next Steps

- [Trends](trends.md) - Learn about piecewise linear trends with changepoint detection
- [Seasonality](seasonality.md) - Configure yearly and weekly patterns
- [Auto-Regression](autoregression.md) - Capture dependencies on recent values
