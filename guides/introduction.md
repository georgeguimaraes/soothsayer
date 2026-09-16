# Introduction to Soothsayer 🧙🔮

Soothsayer is an Elixir library for time series forecasting, inspired by [Facebook's Prophet](https://facebook.github.io/prophet/) and [NeuralProphet](https://neuralprophet.com/).

## The model

A series is a sum of components, each a small network trained together with the others:

```
y(t) = trend(t) + seasonality(t) + ar(t) + events(t) + regressors(t)
```

- trend(t) is the long-term direction, piecewise linear
- seasonality(t) is the yearly, weekly and daily cycles
- ar(t) is what the last few values say about the next one
- events(t) is holidays and promotions on known dates
- regressors(t) is other columns you know ahead of time

Every component can be turned off, and prediction returns each one separately, so you can see what drives a forecast. The networks are [Axon](https://hexdocs.pm/axon), the data goes through [Explorer](https://hexdocs.pm/explorer) and [Nx](https://hexdocs.pm/nx), and if you've used Prophet or NeuralProphet the configuration will look familiar.

## Compared with NeuralProphet

What's here and what isn't, against [NeuralProphet's feature list](https://neuralprophet.com/contents.html):

| Feature | Soothsayer | NeuralProphet |
|---------|------------|---------------|
| Trend | Yes | Yes |
| Changepoints | Yes | Yes |
| Yearly seasonality | Yes | Yes |
| Weekly seasonality | Yes | Yes |
| Daily seasonality | Yes | Yes |
| Sub-daily data (hourly, 5-minute, ...) with inferred frequency | Yes | Yes |
| Auto-regression (AR) | Yes | Yes |
| Deep AR-Net | Yes | Yes |
| Direct multi-step forecasting (`n_forecasts`) | Yes | Yes |
| Rolling-origin backtest / validation metrics | Yes | Yes |
| Learning rate range test | Yes | Yes |
| One-cycle learning rate schedule | Yes | Yes |
| Auto batch size and epochs | Yes | Yes |
| Events | Yes | Yes |
| Lagged regressors | Yes | Yes |
| Future regressors | Yes | Yes |
| Country holidays | Yes | Yes |
| Multiplicative events | Planned | Yes |
| Event regularization | Planned | Yes |
| Recurring events | Yes | No |
| Uncertainty estimation | Yes | Yes |
| Multiplicative seasonality | Yes | Yes |

## Quick example

```elixir
alias Explorer.DataFrame
alias Explorer.Series

# Prepare your data with "ds" (dates) and "y" (values) columns
df = DataFrame.new(%{
  "ds" => Date.range(~D[2020-01-01], ~D[2022-12-31]),
  "y" => your_values
})

# Create and fit a model
model = Soothsayer.new()
fitted_model = Soothsayer.fit(model, df)

# Make predictions
future_dates = Series.from_list(Date.range(~D[2023-01-01], ~D[2023-12-31]) |> Enum.to_list())
predictions = Soothsayer.predict(fitted_model, future_dates)
# => a DataFrame with ds, yhat and one column per component:
#    trend, yearly_seasonality, weekly_seasonality, ...

predictions["yhat"]
predictions["trend"]
```

## Next

- [The basics](basics.md), fitting and predicting with trend and seasonality
- [Trends](trends.md), changepoints and regularization
- [Seasonality](seasonality.md), Fourier terms, additive and multiplicative
- [Auto-regression](autoregression.md), lags, multi-step forecasts and the AR network
- [Events](events.md), holidays, promotions and country holidays
- [Regressors](regressors.md), future and lagged regressors
- [Missing data](missing_data.md), what fit does with gaps
- [Uncertainty](uncertainty.md), prediction intervals from quantiles

## Resources

- [NeuralProphet docs](https://neuralprophet.com/contents.html), the Python library this is a port of
- [Prophet docs](https://facebook.github.io/prophet/), where the model family started
- [Livebook tutorial](https://github.com/georgeguimaraes/soothsayer/blob/main/livebook/soothsayer_tutorial.livemd), the examples, runnable
