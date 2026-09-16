# Seasonality

Seasonality is whatever repeats on a fixed period: holiday shopping every December, the weekend dip in signups, the overnight low in an hourly temperature series. Soothsayer models yearly, weekly and daily seasonality with Fourier terms.

## How it works

```
seasonality(t) = sum(a_n * cos(2*pi*n*t/P) + b_n * sin(2*pi*n*t/P))
```

`P` is the period (a year, a week or a day), `n` runs from 1 to the number of Fourier terms, and the `a_n`, `b_n` are learned. More terms give the curve more wiggles to work with.

The seasonal layers have no intercept, so each component is a zero-centered offset around the trend and the trend carries the level of the series. That's also NeuralProphet's setup.

For the math, see [NeuralProphet's seasonality docs](https://neuralprophet.com/html/seasonal-modeling.html).

## Configuration

```elixir
model = Soothsayer.new(%{
  seasonality: %{
    mode: :additive,      # :additive or :multiplicative
    yearly: %{
      enabled: true,
      fourier_terms: 6
    },
    weekly: %{
      enabled: true,
      fourier_terms: 3
    },
    daily: %{
      enabled: :auto,     # on for sub-daily data, off otherwise
      fourier_terms: 6
    }
  }
})
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `:additive` | How seasonality combines with the trend, see below |
| `yearly.enabled` | `true` | Yearly patterns |
| `yearly.fourier_terms` | `6` | Flexibility of the yearly curve |
| `weekly.enabled` | `true` | Weekly patterns |
| `weekly.fourier_terms` | `3` | Flexibility of the weekly curve |
| `daily.enabled` | `:auto` | Time of day patterns |
| `daily.fourier_terms` | `6` | Flexibility of the daily curve |

`enabled` takes `true`, `false` or `:auto`. With `:auto` the decision is made at fit from the data, with the same rules as NeuralProphet: yearly needs at least two years of data, weekly at least two weeks with rows closer than a week apart, daily at least two days with rows closer than a day apart. Daily defaults to `:auto` so daily data is unaffected. Yearly and weekly default to `true` but accept `:auto` too.

## Yearly seasonality

```elixir
model = Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: true, fourier_terms: 6},
    weekly: %{enabled: false}
  }
})
```

Three terms give a smooth curve with a single peak, the default six handles most series, and ten or more is for shapes with several peaks a year. Every extra term is two more coefficients that can fit noise, so go up only when the residuals still show a yearly pattern.

## Weekly seasonality

```elixir
model = Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: false},
    weekly: %{enabled: true, fourier_terms: 3}
  }
})
```

Two terms are enough for a plain weekday versus weekend split. The default three fits most day of week patterns. Go to five or more when individual days behave differently, say a Wednesday spike on top of the weekend dip.

## Daily seasonality

For data with more than one row per day (hourly readings, 15-minute meter data, 5-minute sensor logs):

```elixir
model = Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: false},
    weekly: %{enabled: true},
    daily: %{enabled: true, fourier_terms: 6}
  }
})
```

The Fourier terms run over the fraction of the day that has passed, so midnight is 0 and noon is 0.5. On daily data every row sits at midnight and the daily component has nothing to learn, which is why it defaults to `:auto` and stays off there.

Yearly and weekly seasonality also see the time of day on sub-daily data: a reading at noon on a Wednesday is 3.5 days into the week, not 3. At midnight the values are the same as for a plain date, so switching a daily dataset from dates to datetimes changes nothing.

## Additive vs multiplicative

By default the seasonal effect is added to the trend, so a summer peak is the same size in year one and year five:

```
y(t) = trend(t) + seasonality(t)
```

Many series don't behave like that. Airline passengers, retail sales and web traffic tend to have seasonal swings that grow with the level. For those, use multiplicative mode, where the seasonal effect is a fraction of the trend:

```
y(t) = trend(t) * (1 + seasonality(t))
```

```elixir
model = Soothsayer.new(%{
  seasonality: %{mode: :multiplicative}
})
```

A quick way to choose: plot the series. If the peaks and troughs get wider as the level rises, go multiplicative. If they stay the same width, stay additive.

`Soothsayer.predict/3` and `Soothsayer.predict_components/3` return the seasonal components in absolute units in both modes, the amount added to the trend on each date, so the components always sum to the forecast.

Multiplicative mode only applies to seasonality. Events, regressors and auto-regression stay additive.

## Choosing Fourier terms

More terms mean more flexibility and more room to overfit. Add terms when the residuals still show a repeating pattern, or when you know the seasonal shape has several peaks per cycle. Remove terms when the fitted curve looks jagged, or when the training fit is good but the holdout isn't. Defaults first, then adjust.

## Example: extracting seasonal components

```elixir
alias Explorer.DataFrame
alias Explorer.Series

# Data with yearly and weekly seasonality
start_date = ~D[2020-01-01]
dates = Date.range(start_date, ~D[2022-12-31])

y = Enum.map(dates, fn date ->
  days = Date.diff(date, start_date)
  trend = 1000
  yearly = 50 * :math.sin(2 * :math.pi() * days / 365.25)
  weekly = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
  noise = :rand.normal(0, 10)
  trend + yearly + weekly + noise
end)

df = DataFrame.new(%{"ds" => dates, "y" => y})

model = Soothsayer.new(%{
  trend: %{changepoints: 0},
  seasonality: %{
    yearly: %{enabled: true, fourier_terms: 6},
    weekly: %{enabled: true, fourier_terms: 3}
  },
  epochs: 50
})

fitted = Soothsayer.fit(model, df)

components = Soothsayer.predict_components(fitted, df["ds"])

# components.yearly_seasonality is the yearly pattern
# components.weekly_seasonality is the weekly pattern
# components.daily_seasonality is zero here, the data is daily
```

`Soothsayer.predict/3` gives the same numbers as a DataFrame with `yearly_seasonality` and `weekly_seasonality` columns next to `yhat` and `trend`. Disabled components don't get a column.

## Disabling seasonality

For a series with no seasonal pattern:

```elixir
model = Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: false},
    weekly: %{enabled: false}
  }
})
```

Or keep just one:

```elixir
model = Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: true},
    weekly: %{enabled: false}
  }
})
```

## Data requirements

Yearly seasonality needs at least two years of data to learn something reliable, weekly needs a few weeks, and daily needs sub-daily data spanning at least a couple of days. With less than that the model learns noise, so disable the component or use `enabled: :auto` and let the fit decide.

## Network architecture

Each enabled period adds `2 * fourier_terms` input features, a sin and a cos per term:

```elixir
yearly_features = 2 * 6  # 12 features for fourier_terms: 6
weekly_features = 2 * 3  # 6 features for fourier_terms: 3
daily_features = 2 * 6   # 12 features for fourier_terms: 6
```

## Next steps

- [Auto-regression](autoregression.md) for dependencies on recent values
- [Trends](trends.md) for piecewise linear trends with changepoints
