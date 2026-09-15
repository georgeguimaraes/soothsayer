# Future Regressors

A future regressor is an external variable that helps explain your series and whose value you know for the dates you want to forecast. Temperature from a weather forecast, planned marketing spend, a published price schedule, the number of stores open. If you know it ahead of time, it can go in.

This is different from auto-regression, which uses the series' own past. Regressors bring in outside information the date alone can't provide.

## How It Works

Each regressor is one input column feeding a linear layer, added to the forecast alongside trend, seasonality and events:

```
y(t) = trend(t) + seasonality(t) + events(t) + sum(w_i * regressor_i(t))
```

Regressor values are z-scored using the training data, so the learned weight `w_i` reads as the effect of one standard deviation of the regressor on the normalized target.

## Configuration

Name the columns you want to use:

```elixir
model = Soothsayer.new(%{
  regressors: ["temperature", "marketing_spend"]
})
```

The training dataframe must contain those columns next to `ds` and `y`:

```elixir
df = DataFrame.new(%{
  "ds" => dates,
  "y" => sales,
  "temperature" => temperatures,
  "marketing_spend" => spend
})

fitted = Soothsayer.fit(model, df)
```

Fitting raises if a configured column is missing.

## Predicting

Prediction needs the regressor values for every date you ask for. Pass them as a dataframe with `ds` and the regressor columns:

```elixir
future_regressors = DataFrame.new(%{
  "ds" => future_dates,
  "temperature" => forecast_temperatures,
  "marketing_spend" => planned_spend
})

predictions = Soothsayer.predict(fitted, Series.from_list(future_dates), regressors: future_regressors)
```

Soothsayer raises if a date has no regressor row rather than filling in zeros, since a forecast built on a made-up regressor value would be wrong without saying so.

When auto-regression is also enabled, the regressors dataframe must cover the days between the last observation and your forecast dates too. Those days get predicted on the way, see the [Auto-Regression guide](autoregression.md). It doesn't need to repeat the training period, the model remembers those values, and it doesn't need the days of the last forecast block past your latest date, which are filled with the training mean since they only affect their own day.

## Inspecting Coefficients

```elixir
Soothsayer.get_regressor_effects(fitted)
# => %{"temperature" => 0.42, "marketing_spend" => 0.18}
```

Positive means the regressor pushes the forecast up. The `regressors` column of `Soothsayer.predict/3` (the `:regressors` key of `Soothsayer.predict_components/3`) holds the combined regressor contribution per date in the units of `y`.

## Example: Energy Price and Temperature

The benchmark suite fits NeuralProphet's daily energy price dataset with 14 auto-regressive lags and temperature as a future regressor:

```elixir
model = Soothsayer.new(%{
  ar: %{enabled: true, lags: 14},
  trend: %{changepoints: 0},
  regressors: ["temperature"]
})

fitted = Soothsayer.fit(model, train)

Soothsayer.predict(fitted, validation["ds"],
  history: validation,     # observed prices seed the lags
  regressors: validation   # observed temperatures
)
```

## Lagged Regressors

Sometimes it's the regressor's past that matters: yesterday's temperature for today's energy price, last week's ad spend for this week's sales. Lagged regressors feed the last `lags` values of a column into the forecast, the way auto-regression feeds the target's own past:

```elixir
model = Soothsayer.new(%{
  ar: %{enabled: true, lags: 14},
  lagged_regressors: %{"temperature" => %{lags: 3}}
})
```

They need auto-regression enabled, since their lag windows are built from the same forecast origins. The longest lag among the AR component and the lagged regressors decides where training can start.

The same column can be both a future regressor (its value on the forecast date) and a lagged regressor (its values before the origin). The energy benchmark does exactly that with temperature, matching NeuralProphet's configuration.

### Predicting with lagged regressors

Lagged regressors are read only up to each forecast origin, never on the dates being forecast, and the training values are remembered on the model. So predicting the first block after the training data needs nothing extra. Going further needs the regressor's values for the days in between, passed in the same `regressors:` dataframe:

```elixir
newer = DataFrame.new(%{"ds" => recent_dates, "temperature" => recent_temperatures})
Soothsayer.predict(fitted, Series.from_list(future_dates), regressors: newer)
```

Soothsayer raises naming the first missing date rather than filling in zeros. The `lagged_regressors` column of `Soothsayer.predict/3` holds their combined contribution.

## Not Yet Supported

- **Hidden layers for lagged regressors** (NeuralProphet's `lagged_reg_layers`). Lagged regressors are linear.
- **Multiplicative regressors** that scale with the trend. Regressors are always additive today.

## Next Steps

- [Events](events.md) - One-off and recurring dates
- [Auto-Regression](autoregression.md) - Using the series' own past
- [The Basics](basics.md) - Fundamental concepts
