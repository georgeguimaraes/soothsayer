# Soothsayer 🧙🔮

[![Run in Livebook](https://livebook.dev/badge/v1/blue.svg)](https://livebook.dev/run?url=https%3A%2F%2Fgithub.com%2Fgeorgeguimaraes%2Fsoothsayer%2Fblob%2Fmain%2Flivebook%2Fsoothsayer_tutorial.livemd)

Soothsayer is an Elixir library for time series forecasting, inspired by Facebook's Prophet and NeuralProphet. It decomposes your time series into interpretable components (trend, seasonality, auto-regression, events) and uses neural networks to learn the patterns.

**Warning:** Soothsayer is currently in alpha stage. The API is unstable and may change at any moment without prior notice. Use with caution in production environments.

## Installation

Add `soothsayer` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:soothsayer, "~> 0.1.0"},
  ]
end
```

Then run `mix deps.get` to install the dependencies.

## Quick Start

```elixir
alias Explorer.DataFrame
alias Explorer.Series

# Your data needs two columns: "ds" (dates) and "y" (values)
df = DataFrame.new(%{
  "ds" => Date.range(~D[2020-01-01], ~D[2022-12-31]),
  "y" => your_values
})

# Create and fit the model
model = Soothsayer.new()
fitted_model = Soothsayer.fit(model, df)

# Make predictions
future_dates = Date.range(~D[2023-01-01], ~D[2023-12-31])
predictions = Soothsayer.predict(fitted_model, Series.from_list(Enum.to_list(future_dates)))
```

You can also get individual components to understand what's driving the forecast:

```elixir
components = Soothsayer.predict_components(fitted_model, future_dates_series)
# => %{combined: ..., trend: ..., yearly_seasonality: ..., weekly_seasonality: ..., ar: ..., events: ...}
```

To model special events like holidays or promotions:

```elixir
# Define events with optional windows (days before/after)
model = Soothsayer.new(%{
  events: %{"black_friday" => %{lower_window: -1, upper_window: 1}}
})

events_df = DataFrame.new(%{
  "event" => ["black_friday", "black_friday"],
  "ds" => [~D[2021-11-26], ~D[2022-11-25]]
})

fitted_model = Soothsayer.fit(model, df, events: events_df)
```

Click the "Run in Livebook" badge above to try the interactive tutorial, or check the `livebook` directory for examples.

## Features

Soothsayer models time series as a sum of components:

```
y(t) = trend(t) + seasonality(t) + ar(t) + events(t)
```

Each component can be enabled or disabled depending on your data.

### Trend

Captures long-term growth or decline in your data. Enable this when your data has a general upward or downward direction over time.

```elixir
Soothsayer.new(%{
  trend: %{enabled: true}  # this is the default
})
```

Good for: sales growth, user adoption, gradual temperature changes.

#### Changepoints

By default, Soothsayer uses piecewise linear trends with automatic changepoint detection. This allows the trend to change slope at multiple points, capturing shifts in growth rate (e.g., a product launch, market change, or policy update).

```elixir
Soothsayer.new(%{
  trend: %{
    changepoints: 10,      # number of potential changepoints (default: 10)
    changepoints_range: 0.8  # place changepoints in first 80% of data (default: 0.8)
  }
})
```

The model learns which changepoints matter and how much the slope changes at each one. Setting `changepoints: 0` disables changepoints and uses a simple linear trend.

**Trend regularization** can prevent overfitting when you have many changepoints:

```elixir
trend: %{
  changepoints: 25,
  regularization: 0.1  # L1 penalty pushes small slope changes toward zero
}
```

This is useful when you're not sure how many changepoints you need. Set more than you think necessary and let regularization prune the unimportant ones.

### Seasonality

Captures repeating patterns at fixed intervals. Soothsayer supports yearly and weekly seasonality using Fourier terms.

```elixir
Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: true, fourier_terms: 6},
    weekly: %{enabled: true, fourier_terms: 3}
  }
})
```

**Yearly seasonality** captures patterns that repeat every year (holiday shopping, summer peaks, etc). More `fourier_terms` means more flexibility to fit complex seasonal shapes, but also more risk of overfitting.

**Weekly seasonality** captures patterns that repeat every week (weekend dips, Monday spikes, etc). Usually needs fewer fourier terms than yearly.

| fourier_terms | Flexibility | Use when |
|---------------|-------------|----------|
| 3 | Low | Simple, smooth seasonal patterns |
| 6 | Medium | Most cases (default for yearly) |
| 10+ | High | Complex patterns with sharp peaks |

**Multiplicative seasonality:** by default the seasonal effect is added to the trend. If your seasonal swings grow with the level of the series (airline passengers, retail sales), make them a fraction of the trend instead:

```elixir
Soothsayer.new(%{
  seasonality: %{mode: :multiplicative}
})
```

### Auto-Regression (AR)

Captures dependencies on recent values. Enable this when today's value depends on yesterday's (or the last few days). This is common in financial data, sensor readings, and anything with momentum.

```elixir
Soothsayer.new(%{
  ar: %{
    enabled: true,
    lags: 7           # use the last 7 values to predict the next one
  }
})
```

**Choosing `lags`:** Start with the natural cycle of your data. For daily data with weekly patterns, try 7. For data with monthly patterns, try 30. You can also look at autocorrelation plots to see how many lags are actually useful.

**Forecasting ahead:** dates inside the training data use the observed values as lags. Dates past the last observation are forecast one day at a time, with each prediction feeding the next day's lags, so errors compound over long horizons. If you have observations newer than the training data, pass them as `history:` to seed the lags without refitting:

```elixir
Soothsayer.predict(fitted_model, future_dates, history: recent_df)  # recent_df has "ds" and "y"
```

See the [Auto-Regression guide](guides/autoregression.md) for details.

#### Deep AR-Net

For non-linear autoregressive patterns, you can add hidden layers:

```elixir
ar: %{
  enabled: true,
  lags: 7,
  layers: [32, 16]  # two hidden layers with ReLU activation
}
```

Use this when the relationship between past and future values is complex. For simple linear relationships, leave `layers` empty (the default).

#### Regularization

L1 regularization pushes AR weights toward zero, which prevents overfitting when you have many lags:

```elixir
ar: %{
  enabled: true,
  lags: 14,
  regularization: 0.1  # higher = more sparsity
}
```

This is useful when you're not sure how many lags to use. Set a higher `lags` than you think you need and let regularization prevent the model from overfitting to noise in distant lags.

### Events

Captures the impact of special occasions (holidays, promotions, etc.) that affect your time series. Events are modeled as additive effects that spike on specific dates.

```elixir
alias Explorer.DataFrame

# Define which events to model and their windows
model = Soothsayer.new(%{
  events: %{
    "black_friday" => %{lower_window: -1, upper_window: 1},
    "christmas" => %{lower_window: -3, upper_window: 0}
  }
})

# Create a DataFrame with event dates
events_df = DataFrame.new(%{
  "event" => ["black_friday", "black_friday", "christmas", "christmas"],
  "ds" => [~D[2022-11-25], ~D[2023-11-24], ~D[2022-12-25], ~D[2023-12-25]]
})

# Fit with events
fitted_model = Soothsayer.fit(model, df, events: events_df)

# Predict (include future events)
future_events = DataFrame.new(%{
  "event" => ["black_friday", "christmas"],
  "ds" => [~D[2024-11-29], ~D[2024-12-25]]
})
predictions = Soothsayer.predict(fitted_model, future_dates, events: future_events)
```

#### Event Windows

Windows allow events to affect surrounding days, not just the event date itself:

- **`lower_window`**: Days before the event (use negative numbers). `-2` means the effect starts 2 days before.
- **`upper_window`**: Days after the event. `1` means the effect extends 1 day after.

Example: `%{lower_window: -1, upper_window: 1}` creates effects for the day before, the event day, and the day after (3 separate learned coefficients).

#### Getting Event Effects

After training, you can extract the learned impact of each event:

```elixir
effects = Soothsayer.get_event_effects(fitted_model)
# => %{"black_friday_-1" => 12.5, "black_friday_0" => 45.2, "black_friday_+1" => 8.3, ...}
```

This shows how much each event (at each window position) adds to the forecast.

### Training Parameters

```elixir
Soothsayer.new(%{
  epochs: 100,         # passes over the training data (default: 100)
  learning_rate: 0.01, # how fast to learn (default: 0.01)
  batch_size: nil,     # rows per gradient step (default: picked from the data size)
  seed: nil            # integer for reproducible fits (default: random)
})
```

Training runs in shuffled minibatches, so one epoch is one pass over the data. When `batch_size` is `nil`, Soothsayer picks a size from the number of rows (16 for a few hundred rows, 32 for a few thousand, up to 512), the same heuristic NeuralProphet uses. Smaller batches mean more gradient steps per epoch.

If your model is underfitting (predictions are too smooth), try more epochs or a higher learning rate. If it's overfitting (fits training data but not new data), try fewer epochs or more regularization.

### Future Regressors

External variables you know ahead of time, like a temperature forecast or planned marketing spend. Name the columns, include them in the training data, and pass their future values when predicting:

```elixir
model = Soothsayer.new(%{regressors: ["temperature"]})
fitted_model = Soothsayer.fit(model, df)  # df has "ds", "y" and "temperature"

future_regressors = DataFrame.new(%{"ds" => future_dates, "temperature" => forecast_temperatures})
predictions = Soothsayer.predict(fitted_model, future_dates_series, regressors: future_regressors)

Soothsayer.get_regressor_effects(fitted_model)
# => %{"temperature" => 0.42}
```

Prediction raises if any requested date is missing from the regressors dataframe rather than guessing. See the [Regressors guide](guides/regressors.md).

## Using EXLA for Faster Training

Soothsayer uses EXLA for training by default, which compiles to XLA for faster execution on CPU/GPU.

Make sure EXLA is configured as your Nx backend in `config/config.exs`:

```elixir
config :nx, default_backend: EXLA.Backend
```

Or set it at runtime:

```elixir
Nx.global_default_backend(EXLA.Backend)
```

### GPU Memory Configuration

By default, XLA pre-allocates 90% of GPU memory at startup for performance. This can cause issues if you're sharing the GPU with other applications (like X windows, other ML processes, or running multiple notebooks).

If you see errors like `CUDNN_STATUS_INTERNAL_ERROR` or out-of-memory errors when starting, configure EXLA to disable preallocation or limit memory usage in `config/config.exs`:

```elixir
# Disable preallocation (allocates on-demand)
config :exla, :clients,
  cuda: [platform: :cuda, preallocate: false]

# Or limit to 50% of GPU memory
config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.5]
```

Alternatively, set environment variables before starting your application:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Or: export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

| Option | Effect |
|--------|--------|
| `preallocate: false` | Allocates memory on-demand instead of upfront |
| `memory_fraction: 0.5` | Pre-allocates only 50% of GPU memory |

## Full Configuration Example

Here's a model configured for daily sales data with yearly seasonality, short-term momentum, and holiday effects:

```elixir
model = Soothsayer.new(%{
  trend: %{enabled: true, changepoints: 10},
  seasonality: %{
    yearly: %{enabled: true, fourier_terms: 8},
    weekly: %{enabled: true, fourier_terms: 3}
  },
  ar: %{
    enabled: true,
    lags: 7,
    regularization: 0.05
  },
  events: %{
    "black_friday" => %{lower_window: -1, upper_window: 1},
    "christmas" => %{lower_window: -3, upper_window: 0}
  },
  epochs: 150,
  learning_rate: 0.01
})
```

## Benchmarks Against NeuralProphet

The test suite includes a benchmark layer that fits Soothsayer on the datasets NeuralProphet uses in its own model performance tests, with the same 90/10 split, and prints validation metrics next to the numbers NeuralProphet's CI publishes. It's excluded from the default run:

```bash
mix test --only benchmark
```

Results as of September 2026 (lower is better). The Soothsayer column is the benchmark's fixed seed, the range is over six seeds:

| Dataset | Metric | NeuralProphet | Soothsayer | Range over seeds | Notes |
|---------|--------|---------------|------------|------------------|-------|
| Peyton Manning (daily) | MAE | 0.350 | 0.286 | 0.29 to 0.35 | identical configuration |
| Peyton Manning (daily) | RMSE | 0.501 | 0.473 | 0.47 to 0.54 | identical configuration |
| Energy price (daily, AR 14 lags + temperature) | MAE | 5.40 | 4.96 | 4.80 to 5.28 | one step ahead vs NeuralProphet's 7-step average; temperature adds nothing one step ahead (4.66 to 5.19 without it) |
| Energy price (daily, AR 14 lags + temperature) | RMSE | 6.71 | 6.31 | 6.06 to 6.71 | same caveat |
| Air passengers (monthly, multiplicative) | MAE | 30.1 | 23.1 | 22.0 to 30.7 | identical configuration, 130 training rows so the seed matters |
| Air passengers (monthly, multiplicative) | RMSE | 31.1 | 25.0 | 24.1 to 33.2 | same caveat |

The datasets live in `test/fixtures/neuralprophet/` under NeuralProphet's MIT license. NeuralProphet's Yosemite benchmark (5-minute data) is not included since Soothsayer only supports daily dates today.

## Features Not Yet Implemented

The following NeuralProphet features are on the roadmap:

- Lagged Regressors (past values of an external variable, like auto-regression on another series)
- Country Holidays (automatic holiday detection via `:holidefs` library)
- Multiplicative Events (events that scale with trend)
- Event Regularization
- Recurring Events (auto-expand to all years)
- Uncertainty Estimation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automated releases. See the [release documentation](https://github.com/georgeguimaraes/soothsayer/blob/main/.github/RELEASE.md) for details.

## License

Copyright 2024 George Guimarães

Soothsayer is released under the Apache License 2.0. See the LICENSE file for details.
