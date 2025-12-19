# Introduction to Soothsayer

Soothsayer is an Elixir library for time series forecasting, inspired by [Facebook's Prophet](https://facebook.github.io/prophet/) and [NeuralProphet](https://neuralprophet.com/).

## What is Soothsayer?

Soothsayer decomposes your time series into interpretable components and uses neural networks to learn the patterns. The model equation is:

```
y(t) = trend(t) + seasonality(t) + ar(t)
```

Where:
- **trend(t)** captures long-term growth or decline
- **seasonality(t)** captures repeating patterns (yearly, weekly)
- **ar(t)** captures dependencies on recent values (auto-regression)

Each component can be enabled or disabled depending on your data.

## Why Soothsayer?

- **Interpretable**: See what each component contributes to the forecast
- **Neural network powered**: Uses [Axon](https://hexdocs.pm/axon) for flexible learning
- **Elixir native**: Built for the BEAM ecosystem with [Nx](https://hexdocs.pm/nx) and [Explorer](https://hexdocs.pm/explorer)
- **Familiar API**: If you've used Prophet or NeuralProphet, you'll feel at home

## Comparison with NeuralProphet

Soothsayer implements a subset of [NeuralProphet's features](https://neuralprophet.com/contents.html):

| Feature | Soothsayer | NeuralProphet |
|---------|------------|---------------|
| Trend | Yes | Yes |
| Changepoints | Yes | Yes |
| Yearly seasonality | Yes | Yes |
| Weekly seasonality | Yes | Yes |
| Auto-regression (AR) | Yes | Yes |
| Deep AR-Net | Yes | Yes |
| Lagged regressors | Planned | Yes |
| Future regressors | Planned | Yes |
| Events/Holidays | Planned | Yes |
| Uncertainty quantification | Planned | Yes |

## Quick Example

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

# Get individual components
components = Soothsayer.predict_components(fitted_model, future_dates)
# => %{combined: ..., trend: ..., yearly_seasonality: ..., weekly_seasonality: ...}
```

## Next Steps

- [The Basics](basics.md) - Learn the fundamentals with trend and seasonality
- [Trends](trends.md) - Piecewise linear trends with changepoint detection
- [Seasonality](seasonality.md) - Yearly and weekly patterns with Fourier terms
- [Auto-Regression](autoregression.md) - Capture dependencies on recent values

## Resources

- [NeuralProphet Documentation](https://neuralprophet.com/contents.html) - The Python library that inspired Soothsayer
- [Prophet Documentation](https://facebook.github.io/prophet/) - Facebook's original forecasting library
- [Interactive Livebook Tutorial](https://github.com/georgeguimaraes/soothsayer/blob/main/livebook/soothsayer_tutorial.livemd) - Run the examples yourself
