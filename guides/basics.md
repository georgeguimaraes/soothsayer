# The basics

How to get data in, fit a model, predict and check the result.

## Data format

Soothsayer expects an Explorer DataFrame with two columns:

- `ds`: timestamps, a `:date` or `{:naive_datetime, _}` series, strictly increasing
- `y`: the values to forecast, numeric

```elixir
alias Explorer.DataFrame

df = DataFrame.new(%{
  "ds" => [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]],
  "y" => [100.0, 102.5, 101.3]
})
```

Sub-daily data works the same way with naive datetimes. When reading a CSV, give Explorer the dtype so the column doesn't come in as strings:

```elixir
df = DataFrame.from_csv!("readings.csv", dtypes: [{"ds", {:naive_datetime, :microsecond}}])
```

The step between rows (the frequency) is inferred at fit from the most common gap in `ds`, so daily, hourly, 5-minute and monthly data all work without configuration. It can also be set explicitly with `frequency: {5, :minute}` (units `:minute`, `:hour`, `:day`, `:month`). Auto-regression lags, forecast blocks and event windows all move by that step.

Missing values and missing rows are fine: fit drops or imputes them the way NeuralProphet does and logs what it did. See [Missing Data](missing_data.md).

## Creating a model

`Soothsayer.new/1` takes the configuration as a map:

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

### Default configuration

`Soothsayer.new()` without arguments uses these defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trend.enabled` | `true` | Enable trend component |
| `trend.changepoints` | `10` | Number of potential changepoints, or a list of dates to bend at |
| `trend.changepoints_range` | `0.8` | Place changepoints in first 80% of data |
| `trend.growth` | `:linear` | `:discontinuous` lets the trend jump at changepoints, see [Trends](trends.md) |
| `trend.regularization` | `nil` | L1 penalty on the slope changes |
| `seasonality.mode` | `:additive` | `:additive` or `:multiplicative` seasonality |
| `seasonality.regularization` | `nil` | L1 penalty on every seasonal coefficient |
| `seasonality.yearly.enabled` | `true` | Enable yearly seasonality |
| `seasonality.yearly.fourier_terms` | `6` | Flexibility of yearly pattern |
| `seasonality.weekly.enabled` | `true` | Enable weekly seasonality |
| `seasonality.weekly.fourier_terms` | `3` | Flexibility of weekly pattern |
| `seasonality.daily.enabled` | `:auto` | Daily seasonality, on for sub-daily data with at least two days of it |
| `seasonality.daily.fourier_terms` | `6` | Flexibility of daily pattern |
| `seasonality.custom` | `%{}` | Other periods, `%{"monthly" => %{period: 30.5, fourier_terms: 3}}`; any period takes a `condition` column, see [Seasonality](seasonality.md) |
| `frequency` | `:auto` | Step between rows, inferred from `ds`, or `{amount, unit}` such as `{1, :hour}` |
| `ar.enabled` | `false` | Auto-regression on the series' own past, with `ar.lags`, see [Auto-regression](autoregression.md) |
| `ar.layers` | `[]` | Hidden layers of the AR network, empty means linear |
| `ar.regularization` | `nil` | L1 penalty on the AR weights |
| `ar.forecast_steps` | `1` | Steps ahead the AR head forecasts directly, NeuralProphet's `n_forecasts` |
| `events` | `%{}` | Event name to `%{steps_before, steps_after, mode, regularization, recurring}`, see [Events](events.md) |
| `holidays` | no countries | `countries`, `steps_before`, `steps_after`, `mode`, `regularization`, `types`, `language`, see [Events](events.md) |
| `regressors` | `%{}` | Column name to `%{mode, regularization, layers}`, a list of names also works, see [Regressors](regressors.md) |
| `lagged_regressors` | `%{}` | Column name to `%{lags: n}` for lagged regressors, needs AR |
| `lagged_regressors_layers` | `[]` | Hidden layers of one shared network over all lagged regressors |
| `missing` | impute | `impute`, `impute_linear: 10`, `impute_rolling: 10`, `drop_samples: false`, see [Missing data](missing_data.md) |
| `quantiles` | `[]` | Prediction interval quantiles, e.g. `[0.1, 0.9]`, see [Uncertainty](uncertainty.md) |
| `epochs` | `:auto` | Passes over the training data, picked from the data size, or a number |
| `learning_rate` | `:auto` | Found by a learning rate range test before training, or a number |
| `series` | `%{column: nil, normalize: :local, trend: :global, seasonality: :global, local_regularization: nil, unknown: :error}` | Several series in one model, told apart by `column`, with shared or per-series trend and seasonality, see [Several series](series.md) |
| `recency` | `%{enabled: true, weight: 2, start: 0.0}` | Recent rows weigh more in the loss, the last one `weight` times the oldest, see [Trends](trends.md#recent-data-first) |
| `schedule` | `:one_cycle` | Learning rate schedule, `:one_cycle` or `:constant` |
| `optimizer` | `:adam` | `:adam` or `:adamw` |
| `batch_size` | `nil` | Rows per gradient step, picked from the data size when `nil` |
| `seed` | `nil` | Integer seed for reproducible fits, random when `nil` |

## Fitting the model

`Soothsayer.fit/2` trains the model on your data:

```elixir
fitted_model = Soothsayer.fit(model, df)
```

Training and prediction are compiled with [EXLA](https://hexdocs.pm/exla). Set it as the Nx default backend too, so the tensors you build outside fit live on the same backend:

```elixir
# In config/config.exs
config :nx, default_backend: EXLA.Backend

# Or at runtime
Nx.global_default_backend(EXLA.Backend)
```

### GPU memory configuration

XLA pre-allocates 90% of GPU memory at startup. If the GPU is shared with other processes and you see `CUDNN_STATUS_INTERNAL_ERROR` or out of memory errors, turn preallocation off:

```elixir
# In config/config.exs
config :exla, :clients,
  cuda: [platform: :cuda, preallocate: false]
```

Or with an environment variable before starting:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

See the [README](https://github.com/georgeguimaraes/soothsayer#gpu-memory-configuration) for more options.

## Making predictions

`Soothsayer.predict/2` takes an Explorer Series of dates:

```elixir
alias Explorer.Series

future_dates = Series.from_list([~D[2023-01-04], ~D[2023-01-05], ~D[2023-01-06]])
predictions = Soothsayer.predict(fitted_model, future_dates)
# => #Explorer.DataFrame<
#      Polars[3 x 5]
#      ds date [2023-01-04, 2023-01-05, 2023-01-06]
#      yhat f64 [103.1, 104.2, 105.0]
#      trend f64 [102.9, 103.4, 103.9]
#      yearly_seasonality f64 [0.4, 0.5, 0.6]
#      weekly_seasonality f64 [-0.2, 0.3, 0.5]
#    >
```

The result is a DataFrame with one row per date: `ds` (your series, same dtype), `yhat`, one column per configured quantile (`yhat_10`, `yhat_90`, see [Uncertainty](uncertainty.md)), then `trend` and one column per enabled component. The component columns add up to `yhat`.

## Getting components

The component columns are how you see what drives a forecast, trend or seasonality, and by how much. Components that are disabled in the config have no column. When the trend is disabled its column is a flat line at the training mean, so the columns still add up.

`Soothsayer.predict_components/2` returns the same numbers as a map of tensors, for when you'd rather stay in Nx. Disabled components are zeros there:

```elixir
components = Soothsayer.predict_components(fitted_model, future_dates)
# => %{
#   combined: #Nx.Tensor<...>,
#   trend: #Nx.Tensor<...>,
#   yearly_seasonality: #Nx.Tensor<...>,
#   weekly_seasonality: #Nx.Tensor<...>,
#   daily_seasonality: #Nx.Tensor<...>,
#   ar: #Nx.Tensor<...>,
#   ...
#   quantiles: %{}
# }
```

## Complete example

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
```

## Visualizing results

In Livebook, VegaLite does the job:

```elixir
alias VegaLite, as: Vl

df_with_predictions = DataFrame.put(df, "yhat", predictions["yhat"])

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

The [Livebook tutorial](https://github.com/georgeguimaraes/soothsayer/blob/main/livebook/soothsayer_tutorial.livemd) has more plots.

## Evaluating a configuration

Before trusting a configuration, backtest it. `Soothsayer.backtest/3` holds out the last part of the data (10% by default), fits on the rest, and forecasts `horizon` steps ahead from every held out origin using only what was observed up to it. `horizon` defaults to the model's `ar.forecast_steps`, or 1 without auto-regression:

```elixir
result = Soothsayer.backtest(model, df, horizon: 7, validation_fraction: 0.1)

result.metrics.mean_absolute_error
result.by_step[1].mean_absolute_error   # one day ahead
result.by_step[7].mean_absolute_error   # a week ahead
```

`result.predictions` is a dataframe of every forecast with columns `origin`, `ds`, `step`, `y` and `yhat`, so you can plot errors by horizon or by season. Events and regressors go in as `events:` and `regressors:` options.

## Next steps

- [Trends](trends.md): piecewise linear trends with changepoints
- [Seasonality](seasonality.md): yearly, weekly and daily patterns
- [Auto-Regression](autoregression.md): dependence on recent values
