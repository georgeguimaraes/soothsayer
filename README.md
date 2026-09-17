# Soothsayer 🧙🔮

[![Run in Livebook](https://livebook.dev/badge/v1/blue.svg)](https://livebook.dev/run?url=https%3A%2F%2Fgithub.com%2Fgeorgeguimaraes%2Fsoothsayer%2Fblob%2Fmain%2Flivebook%2Fsoothsayer_tutorial.livemd)

Soothsayer is an Elixir library for time series forecasting, inspired by Facebook's Prophet and NeuralProphet. It decomposes your time series into interpretable components (trend, seasonality, auto-regression, events) and uses neural networks to learn the patterns.

## Installation

Add `soothsayer` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:soothsayer, "~> 1.0"}
  ]
end
```

Then run `mix deps.get` to install the dependencies.

Soothsayer accepts Nx 0.13 or 1.0. The current Axon release still declares `nx ~> 0.10`, so you get 0.13 by default. To run on Nx 1.0 until Axon ships a release that allows it, add Nx to your own deps with an override:

```elixir
{:nx, "~> 1.0", override: true}
```

## Quick start

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
# => #Explorer.DataFrame<[365 x 5] ds, yhat, trend, yearly_seasonality, weekly_seasonality>
```

The result is a DataFrame: `ds`, `yhat`, a column per configured quantile and one per enabled component, which add up to `yhat`. So what drives the forecast is right there:

```elixir
predictions["yhat"]                 # the forecast
predictions["trend"]                # what the trend contributes
predictions["yearly_seasonality"]   # and the yearly pattern

DataFrame.put(df, "yhat", Soothsayer.predict(fitted_model, df["ds"])["yhat"])  # next to the actuals
```

`Soothsayer.predict_components/3` returns the same values as a map of tensors, `%{combined: ..., trend: ..., quantiles: %{...}}`.

To model special events like holidays or promotions:

```elixir
# Define events with optional windows (steps before/after, days here)
model = Soothsayer.new(%{
  events: %{"black_friday" => %{steps_before: 1, steps_after: 1}}
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
y(t) = trend(t) + seasonality(t) + ar(t) + events(t) + regressors(t)
```

Each component can be enabled or disabled depending on your data.

### Trend

The long-term direction of the series, on by default:

```elixir
Soothsayer.new(%{
  trend: %{enabled: true}
})
```

#### Changepoints

The trend is piecewise linear. It can change slope at up to `changepoints` points spread over the first `changepoints_range` of the data, and each segment learns its own slope, so a product launch or a market shift shows up as a bend.

```elixir
Soothsayer.new(%{
  trend: %{
    changepoints: 10,      # number of potential changepoints (default: 10)
    changepoints_range: 0.8  # place changepoints in first 80% of data (default: 0.8)
  }
})
```

`changepoints: 0` gives a plain linear trend.

Trend regularization is an L1 penalty on the slope changes, for when you have many changepoints:

```elixir
trend: %{
  changepoints: 25,
  regularization: 0.1
}
```

Set more changepoints than you think you need and let the penalty zero out the ones that don't matter.

By default the trend is continuous. `growth: :discontinuous` lets it jump at each changepoint too, one learned intercept per segment, for series with a level shift that no slope change explains. NeuralProphet's `growth: "off"` is `trend: %{enabled: false}` here.

### Seasonality

Repeating patterns, modelled with Fourier terms.

```elixir
Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: true, fourier_terms: 6},
    weekly: %{enabled: true, fourier_terms: 3},
    daily: %{enabled: :auto, fourier_terms: 6}
  }
})
```

More `fourier_terms` fit sharper shapes and overfit more easily. Yearly patterns (holiday shopping, summer peaks) usually need more terms than weekly ones (weekend dips). Daily seasonality is for sub-daily data. `enabled: :auto` turns it on when rows are closer than a day apart and span at least two days, NeuralProphet's rule, and yearly and weekly accept `:auto` too.

| fourier_terms | Use when |
|---------------|----------|
| 3 | smooth patterns (default for weekly) |
| 6 | most cases (default for yearly) |
| 10+ | sharp peaks |

By default the seasonal effect is added to the trend. If the swings grow with the level of the series (airline passengers, retail sales), make them a fraction of the trend instead:

```elixir
Soothsayer.new(%{
  seasonality: %{mode: :multiplicative}
})
```

Any other period goes under `custom`, in days, and a `condition` names a 0 to 1 column in the data that switches a pattern on and off, so a weekly cycle can exist only in summer:

```elixir
Soothsayer.new(%{
  seasonality: %{
    weekly: %{enabled: true, condition: "summer"},
    custom: %{"monthly" => %{period: 30.5, fourier_terms: 3}}
  }
})
```

The condition column travels with the regressors at predict. `seasonality: %{regularization: 0.1}` puts an L1 penalty on every seasonal coefficient. See the [Seasonality guide](guides/seasonality.md).

### Auto-regression

For series where today depends on the last few values, like sensor readings or anything with momentum.

```elixir
Soothsayer.new(%{
  ar: %{
    enabled: true,
    lags: 7,           # use the last 7 values
    forecast_steps: 1  # steps ahead forecast directly from them
  }
})
```

The AR network sees the lags minus the trend, seasonality, events and regressors at each lag's timestamp, as NeuralProphet does, so it models what those components leave over instead of absorbing the level and the cycles itself.

Start `lags` at the natural cycle of the data: 7 for daily data with a weekly pattern, 30 for monthly patterns. An autocorrelation plot tells you how many lags carry signal.

Dates inside the training data use the observed values as lags. Dates past the last observation are forecast in blocks of `forecast_steps` directly from the last real values, then the next block from those predictions, and so on. With the default `forecast_steps: 1` that is one day at a time with errors compounding; set `forecast_steps: 7` to learn a separate weight vector for each of the next 7 days, NeuralProphet's `n_forecasts`. If you have observations newer than the training data, pass them as `history:` to seed the lags without refitting:

```elixir
Soothsayer.predict(fitted_model, future_dates, history: recent_df)  # recent_df has "ds" and "y"
```

See the [Auto-Regression guide](guides/autoregression.md) for details.

#### Deep AR-Net

Hidden layers make the AR part a small network instead of a linear map:

```elixir
ar: %{
  enabled: true,
  lags: 7,
  layers: [32, 16]  # two hidden layers with ReLU activation
}
```

Leave `layers` empty (the default) for linear AR.

#### Regularization

An L1 penalty on the AR weights, for when you use many lags:

```elixir
ar: %{
  enabled: true,
  lags: 14,
  regularization: 0.1  # higher = more sparsity
}
```

Set more lags than you think you need and let the penalty quiet the distant ones.

### Events

Holidays, promotions, anything that lands on known dates. Each event adds a learned amount on its dates, and on the steps around them if you give it a window.

```elixir
alias Explorer.DataFrame

# Define which events to model and their windows
model = Soothsayer.new(%{
  events: %{
    "black_friday" => %{steps_before: 1, steps_after: 1},
    "christmas" => %{steps_before: 3, steps_after: 0}
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

#### Event windows

`steps_before` and `steps_after` extend the effect around the event date, in steps of the data's frequency: `steps_before: 2` starts two days early on daily data and two hours early on hourly data. Both default to `0`, so `%{}` is the event date alone, and `%{steps_before: 1, steps_after: 1}` learns three coefficients, one for the day before, the day itself and the day after.

An event with `mode: :multiplicative` scales with the trend instead of adding a fixed amount, so a promo that lifts sales by 30% stays 30% as the series grows. `regularization: 0.1` puts an L1 penalty on that event's coefficients, for events you suspect do nothing:

```elixir
events: %{
  "promo" => %{steps_before: 1, steps_after: 1, mode: :multiplicative},
  "maybe" => %{regularization: 0.1}
}
```

#### Country holidays and recurring events

Every holiday of a country becomes an event of its own, dates generated for the years in your data and the years you forecast. The dates come from [dayoff](https://hex.pm/packages/dayoff), 200+ countries with their states and regions, no setup needed:

```elixir
model = Soothsayer.new(%{
  holidays: %{countries: ["US"], steps_before: 1, steps_after: 1}
})

fitted_model = Soothsayer.fit(model, df)
Soothsayer.get_event_effects(fitted_model)
# => %{"Christmas Day_0" => 48.5, "Independence Day_0" => 31.2, "Thanksgiving Day_0" => ...}
```

An event that falls on the same month and day every year can be given once with `recurring: :yearly`, and the occurrences given at fit are remembered, so predicting inside the training period or into future years needs no events dataframe. `holidays` takes `mode` and `regularization` too, one value for every holiday. See the [Events guide](guides/events.md).

#### Event effects

After fit, the learned amount for each event and offset:

```elixir
effects = Soothsayer.get_event_effects(fitted_model)
# => %{"black_friday_-1" => 12.5, "black_friday_0" => 45.2, "black_friday_+1" => 8.3, ...}
```


### Training parameters

```elixir
Soothsayer.new(%{
  frequency: :auto,      # step between rows, or {amount, unit} like {1, :hour} (default: inferred)
  epochs: :auto,         # passes over the data, or a number (default: from the data size)
  learning_rate: :auto,  # or a number (default: found by a range test)
  schedule: :one_cycle,  # or :constant (default: one-cycle)
  optimizer: :adam,      # or :adamw
  batch_size: nil,       # rows per gradient step (default: from the data size)
  recency: %{weight: 2, start: 0.0},  # recent rows weigh more in the loss, nil weight turns it off
  seed: nil              # integer for reproducible fits (default: random)
})
```

`ds` can be a `:date` or a `{:naive_datetime, _}` column, so hourly or 5-minute data works the same as daily. The frequency is inferred from the most common gap between rows and drives auto-regression lags, forecast blocks and event windows.

The defaults follow NeuralProphet. Training runs in shuffled minibatches, so one epoch is one pass over the data, and `batch_size` and `epochs` are picked from the number of rows when left at their defaults: small datasets get more passes, large ones fewer.

With `learning_rate: :auto`, Soothsayer runs a learning rate range test before training: about a hundred steps with the rate climbing from `1.0e-6` to `10`, watching the training loss, and picking the rate where the loss falls fastest. That rate is the peak of the one-cycle schedule, which warms up from a tenth of it, peaks at 30% of training, and cools down to a hundredth by the end. The values actually used are recorded on the fitted model's config.

Too smooth a fit wants more epochs or a fixed higher learning rate. A fit that only works on the training data wants fewer epochs or more regularization.

Recent rows count more than old ones, NeuralProphet's newer samples weight: by default the last training row weighs twice the oldest, with a smooth ramp in between, so a slope that changed recently pulls harder. `recency: %{weight: 5, start: 0.5}` makes the last half of the data count up to five times more, `recency: %{weight: nil}` treats every row the same. See the [Trends guide](guides/trends.md#recent-data-first).

### Future regressors

External variables you know ahead of time, like a temperature forecast or planned marketing spend. Name the columns, include them in the training data, and pass their future values when predicting:

```elixir
model = Soothsayer.new(%{regressors: ["temperature"]})
fitted_model = Soothsayer.fit(model, df)  # df has "ds", "y" and "temperature"

future_regressors = DataFrame.new(%{"ds" => future_dates, "temperature" => forecast_temperatures})
predictions = Soothsayer.predict(fitted_model, future_dates_series, regressors: future_regressors)

Soothsayer.get_regressor_effects(fitted_model)
# => %{"temperature" => 0.42}
```

Prediction raises if any requested date is missing from the regressors dataframe rather than guessing.

A map instead of the list gives each regressor its own options: `mode: :multiplicative` makes its effect a fraction of the trend, `regularization` puts an L1 penalty on its coefficient, and `layers` swaps the coefficient for a small network, for effects that curve:

```elixir
Soothsayer.new(%{
  regressors: %{
    "temperature" => %{layers: [16, 8]},
    "marketing_spend" => %{mode: :multiplicative, regularization: 0.1}
  }
})
```

When it's the regressor's past that matters, make it a lagged regressor alongside auto-regression. `lagged_regressors_layers` puts one shared network over all their lags:

```elixir
Soothsayer.new(%{
  ar: %{enabled: true, lags: 14},
  lagged_regressors: %{"temperature" => %{lags: 3}},  # yesterday's and the two days before
  lagged_regressors_layers: [32, 16]
})
```

See the [Regressors guide](guides/regressors.md).

### Missing data

Gaps in the data are handled at fit the way NeuralProphet does. Without auto-regression the rows with a missing `y` (nil or NaN) are dropped. With auto-regression the data is put on the frequency grid, trailing gaps are dropped and the rest are imputed: linearly up to 10 values from each side of a gap, then with a rolling mean over 10 more. Whatever is still missing raises, unless you let fit skip the training samples that touch it:

```elixir
Soothsayer.new(%{
  ar: %{enabled: true, lags: 24},
  missing: %{impute_linear: 10, impute_rolling: 10, drop_samples: true}
})
```

Regressor columns are imputed the same way, and so is the `history:` passed to predict. See the [Missing Data guide](guides/missing_data.md).

### Uncertainty

Ask for quantiles and you get prediction intervals next to the median:

```elixir
model = Soothsayer.new(%{quantiles: [0.1, 0.9]})
fitted_model = Soothsayer.fit(model, df)

predictions = Soothsayer.predict(fitted_model, future_dates)
predictions["yhat"]     # median
predictions["yhat_10"]  # lower line of the 80% interval
predictions["yhat_90"]  # upper line
```

Each quantile is a linear head over the same inputs as the components, trained with the pinball loss, so intervals widen where the series is noisier.

Nothing forces those intervals to cover as much as they promise, so calibrate them on data the model hasn't seen:

```elixir
calibrated = Soothsayer.calibrate(fitted_model, calibration_df, alpha: 0.1)

predictions = Soothsayer.predict(calibrated, future_dates)
predictions["yhat_lower"]  # a 90% interval that holds up on new data
predictions["yhat_upper"]
```

This is conformal prediction: the model forecasts the calibration stretch, and the size of its misses there sets the width. `method: :naive` (the default) builds a band of constant width around `yhat`, `method: :cqr` pushes the quantile band out by the right amount. See the [Uncertainty guide](guides/uncertainty.md).

### Several series

Many related series can share one model, NeuralProphet's global modeling. Put them in one frame with a column that names the series:

```elixir
model = Soothsayer.new(%{series: %{column: "store"}})
fitted_model = Soothsayer.fit(model, df)   # df has ds, y and store

future = DataFrame.new(%{"ds" => dates, "store" => stores})
Soothsayer.predict(fitted_model, future)   # ds, store, yhat, ...
```

Every series is scaled and seeded for its lags on its own, and the components are shared, so a series with little history borrows the shape of the others. When they don't follow one shape, give each its own trend or seasonality:

```elixir
Soothsayer.new(%{
  series: %{column: "store", trend: :local, seasonality: :local, local_regularization: 0.1}
})
```

`local_regularization` pulls the per-series kernels toward their average, NeuralProphet's glocal mode. See the [Several series guide](guides/series.md).

### Evaluating a configuration

`Soothsayer.backtest/3` holds out the last 10% of your data, fits on the rest, and forecasts from every held out origin the way you would in production, using only what was observed up to that point:

```elixir
result = Soothsayer.backtest(model, df, horizon: 7)

result.metrics              # %{mean_absolute_error: ..., root_mean_squared_error: ...}, plus coverage with an interval
result.by_step[7]           # the same, for forecasts made 7 days ahead (with auto-regression)
result.predictions          # DataFrame with origin, ds, step, y, yhat
result.model                # the fitted model
```

This is the protocol NeuralProphet uses for its validation metrics, and what the benchmark suite runs.

## EXLA

Training is compiled with EXLA. Set it as the Nx backend too, so the tensor work around it doesn't fall back to the pure Elixir backend. In `config/config.exs`:

```elixir
config :nx, default_backend: EXLA.Backend
```

Or set it at runtime:

```elixir
Nx.global_default_backend(EXLA.Backend)
```

### GPU memory

XLA grabs 90% of GPU memory at startup. If something else shares the GPU and you see `CUDNN_STATUS_INTERNAL_ERROR` or out-of-memory errors, turn preallocation off or cap it in `config/config.exs`:

```elixir
# Disable preallocation (allocates on-demand)
config :exla, :clients,
  cuda: [platform: :cuda, preallocate: false]

# Or limit to 50% of GPU memory
config :exla, :clients,
  cuda: [platform: :cuda, memory_fraction: 0.5]
```

Or through environment variables:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Or: export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

## Full configuration example

A model for daily sales with yearly seasonality, a week of momentum and two events:

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
    "black_friday" => %{steps_before: 1, steps_after: 1},
    "christmas" => %{steps_before: 3, steps_after: 0}
  },
  quantiles: [0.1, 0.9],
  epochs: 150,
  learning_rate: 0.01
})
```

## Benchmarks against NeuralProphet

The test suite includes a benchmark layer that fits Soothsayer on the datasets NeuralProphet uses in its own model performance tests, with the same 90/10 split, and prints validation metrics next to the numbers NeuralProphet's CI publishes. It's excluded from the default run:

```bash
mix test --only benchmark
```

Results as of September 2026 (lower is better). The Soothsayer column is the benchmark's fixed seed, the range is over five seeds:

| Dataset | Metric | NeuralProphet | Soothsayer | Range over seeds | Notes |
|---------|--------|---------------|------------|------------------|-------|
| Peyton Manning (daily) | MAE | 0.350 | 0.298 | 0.298 to 0.299 | identical configuration |
| Peyton Manning (daily) | RMSE | 0.501 | 0.493 | 0.493 to 0.495 | identical configuration |
| Energy price (daily, AR 14 lags, 7 direct steps, temperature as future and lagged regressor) | MAE | 5.40 | 5.39 | 5.37 to 5.42 | identical configuration and metric (average over horizons 1 to 7) |
| Energy price (daily, AR 14 lags, 7 direct steps, temperature as future and lagged regressor) | RMSE | 6.71 | 6.73 | 6.70 to 6.77 | same |
| Yosemite temperatures (every 5 minutes, AR 36 lags, 12 direct steps, daily seasonality) | MAE | 0.573 | 0.482 | 0.480 to 0.484 | same configuration; yearly seasonality off as NeuralProphet's auto rule does on 65 days; 12 NaN readings imputed at fit |
| Yosemite temperatures (every 5 minutes, AR 36 lags, 12 direct steps, daily seasonality) | RMSE | 0.847 | 0.718 | 0.710 to 0.718 | same |
| Air passengers (monthly, multiplicative) | MAE | 30.1 | 25.1 | 22.4 to 25.9 | identical configuration, 130 training rows so the seed matters |
| Air passengers (monthly, multiplicative) | RMSE | 31.1 | 27.0 | 24.5 to 27.9 | same caveat |

The datasets live in `test/fixtures/neuralprophet/`. Three of them are Prophet's example series (Peyton Manning, Yosemite, air passengers, MIT licensed by Facebook) and the energy price one is a cut of a CC0 Kaggle dataset prepared by NeuralProphet, see the NOTICE file there.

## Not implemented yet

From NeuralProphet, still missing here:

- a choice of loss function (it's Huber)
- the data split utilities, `split_df` and the cross-validation splits

## Contributing

Pull requests are welcome. Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) because releases are cut from them, see [RELEASE.md](https://github.com/georgeguimaraes/soothsayer/blob/main/.github/RELEASE.md).

## License

Copyright 2024 George Guimarães

Soothsayer is released under the Apache License 2.0. See the LICENSE file for details.
