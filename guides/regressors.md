# Future regressors

A future regressor is an external variable that helps explain your series and whose value you know for the dates you want to forecast. Temperature from a weather forecast, planned marketing spend, a published price schedule, the number of stores open. If you know it ahead of time, it can go in.

This is different from auto-regression, which uses the series' own past. Regressors bring in outside information the date alone can't provide.

## How it works

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

A list gives every regressor the defaults. To set options per regressor, use a map from column name to options instead:

```elixir
model = Soothsayer.new(%{
  regressors: %{
    "temperature" => %{mode: :multiplicative},
    "marketing_spend" => %{regularization: 0.1, layers: [16, 8]}
  }
})
```

| Option | Default | Description |
|--------|---------|-------------|
| `mode` | `:additive` | `:additive` adds the effect to the forecast, `:multiplicative` scales it with the trend, see [Multiplicative regressors](#multiplicative-regressors) |
| `regularization` | `nil` | L1 penalty on the regressor's coefficient, see [Regularization](#regularization) |
| `layers` | `[]` | Hidden layer sizes for a regressor that needs a curve, not a line, see [Networks on regressors](#networks-on-regressors) |

Both forms normalize to the map with the defaults filled in, which is what the fitted `config.regressors` shows.

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
future = Soothsayer.future_timestamps(fitted, 30)

future_regressors = DataFrame.new(%{
  "ds" => future,
  "temperature" => forecast_temperatures,
  "marketing_spend" => planned_spend
})

predictions = Soothsayer.predict(fitted, future, regressors: future_regressors)
```

Soothsayer raises if a date has no regressor row rather than filling in zeros, since a forecast built on a made-up regressor value would be wrong without saying so.

When auto-regression is also enabled, the regressors dataframe must cover the days between the last observation and your forecast dates too. Those days get predicted on the way, see the [Auto-Regression guide](autoregression.md). It doesn't need to repeat the training period, the model remembers those values, and it doesn't need the days of the last forecast block past your latest date, which are filled with the training mean since they only affect their own day.

## Inspecting coefficients

```elixir
Soothsayer.get_regressor_effects(fitted)
# => %{"temperature" => 0.42, "marketing_spend" => 0.18}
```

Positive means the regressor pushes the forecast up. The `regressors` column of `Soothsayer.predict/3` (the `:regressors` key of `Soothsayer.predict_components/3`) holds the combined regressor contribution per date in the units of `y`, additive and multiplicative regressors together. A regressor with `layers` has no single coefficient and maps to its layers instead, see below.

## Multiplicative regressors

When a regressor's effect grows with the level of the series, temperature moving sales by a percentage rather than a fixed amount, make it multiplicative:

```elixir
model = Soothsayer.new(%{regressors: %{"temperature" => %{mode: :multiplicative}}})
```

The coefficient is then a fraction of the trend per standard deviation of the regressor, and the effect on a date is that fraction times the trend on that date. Additive and multiplicative regressors can share a model: each mode has its own layer (`regressors_dense` and `regressors_multiplicative_dense`) and predict still reports one `regressors` column in the units of `y`. Same as multiplicative events and seasonality, the trend that does the scaling is detached only at the lag positions of an auto-regressive sample. With the trend disabled the scale is the level of the series, so the regressor behaves like an additive one in another unit.

## Regularization

`regularization: 0.1` puts an L1 penalty on that regressor's coefficient, the lambda times its absolute value, applied from the first training step like the `regularization` on `ar` and `trend`. Use it when you throw in many candidate columns and want the ones that don't help to fade out. On a regressor with `layers` the penalty lands on its first hidden layer. NeuralProphet only starts its penalties in the last third of training, so its lambdas don't transfer.

## Networks on regressors

A coefficient can only draw a line through the regressor. When the relation is a curve, sales that peak at a comfortable temperature and drop on both sides, give the regressor a few hidden layers:

```elixir
model = Soothsayer.new(%{regressors: %{"temperature" => %{layers: [16, 8]}}})
```

Each entry is one hidden dense layer with ReLU, followed by a linear output with no bias, and each networked regressor gets its own network over its own column, NeuralProphet's `neural_nets` future regressor model. There is no shared network across regressors, which keeps the per-regressor effect readable. The layers are named `regressor_temperature_dense_0`, `regressor_temperature_dense_1` and `regressor_temperature_dense_out`, and `get_regressor_effects` returns them for that regressor:

```elixir
Soothsayer.get_regressor_effects(fitted)
# => %{
#   "marketing_spend" => 0.18,
#   "temperature" => %{
#     "regressor_temperature_dense_0" => %{kernel: #Nx.Tensor<f32[1][16]>, bias: #Nx.Tensor<f32[16]>},
#     "regressor_temperature_dense_1" => %{kernel: #Nx.Tensor<f32[16][8]>, bias: #Nx.Tensor<f32[8]>},
#     "regressor_temperature_dense_out" => %{kernel: #Nx.Tensor<f32[8][1]>}
#   }
# }
```

A networked regressor can be multiplicative too.

## Example: energy price and temperature

The benchmark suite fits NeuralProphet's daily energy price dataset with 14 auto-regressive lags, 7 direct forecast steps and temperature as a future regressor:

```elixir
model = Soothsayer.new(%{
  ar: %{enabled: true, lags: 14, forecast_steps: 7},
  trend: %{changepoints: 0},
  regressors: ["temperature"]
})

fitted = Soothsayer.fit(model, train)

Soothsayer.predict(fitted, validation["ds"],
  history: validation,     # observed prices seed the lags
  regressors: validation   # observed temperatures
)
```

## Lagged regressors

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
Soothsayer.predict(fitted, future, regressors: newer)
```

Soothsayer raises naming the first missing date rather than filling in zeros. The `lagged_regressors` column of `Soothsayer.predict/3` holds their combined contribution.

### Hidden layers on lagged regressors

By default every lag of every lagged regressor feeds one linear layer. For a curved relation, give them a shared network with `lagged_regressors_layers`, NeuralProphet's `lagged_reg_layers`:

```elixir
model = Soothsayer.new(%{
  ar: %{enabled: true, lags: 14},
  lagged_regressors: %{"temperature" => %{lags: 3}},
  lagged_regressors_layers: [32, 16]
})
```

One network takes all the lag columns of all the lagged regressors at once, so it can also pick up interactions between them. The hidden layers are named `lagged_regressors_dense_0`, `lagged_regressors_dense_1` and so on, before the `lagged_regressors_dense` output.

## Seasonality conditions ride along

A seasonality with a `condition` reads a 0 to 1 column of your data, see [Conditional seasonality](seasonality.md#conditional-seasonality). That column lives next to the regressors: in the training dataframe at fit, and in the same `regressors:` dataframe at predict, even when the model has no regressors. `Soothsayer.backtest/3` hands the validation frame to predict as `regressors:` by default, so it covers conditions too.

## Related guides

- [Events](events.md) for one-off and recurring dates
- [Auto-regression](autoregression.md) for using the series' own past
- [The Basics](basics.md) for the fit and predict flow
