# Soothsayer

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
  epochs: 100,        # training iterations (default: 100)
  learning_rate: 0.01 # how fast to learn (default: 0.01)
})
```

If your model is underfitting (predictions are too smooth), try more epochs or a higher learning rate. If it's overfitting (fits training data but not new data), try fewer epochs or more regularization.

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

## Full Configuration Example

Here's a model configured for daily sales data with yearly seasonality and short-term momentum:

```elixir
model = Soothsayer.new(%{
  trend: %{enabled: true},
  seasonality: %{
    yearly: %{enabled: true, fourier_terms: 8},
    weekly: %{enabled: true, fourier_terms: 3}
  },
  ar: %{
    enabled: true,
    lags: 7,
    regularization: 0.05
  },
  epochs: 150,
  learning_rate: 0.01
})
```

## Features Not Yet Implemented

The following NeuralProphet features are on the roadmap:

- Lagged Regressors (external variables that affect the forecast)
- Future Regressors (known future values like holidays)
- Country Holidays (automatic holiday detection via `:holidefs` library)
- Multiplicative Events (events that scale with trend)
- Event Regularization
- Recurring Events (auto-expand to all years)
- Uncertainty Estimation
- Multiplicative Seasonality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automated releases. See the [release documentation](https://github.com/georgeguimaraes/soothsayer/blob/main/.github/RELEASE.md) for details.

## License

Soothsayer is released under the Apache License 2.0. See the LICENSE file for details.
