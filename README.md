# Soothsayer

Soothsayer is an Elixir library for time series forecasting, inspired by Facebook's Prophet and NeuralProphet. It provides a flexible and easy-to-use interface for creating and training forecasting models.

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

## Usage

Here's a basic example of how to use Soothsayer:

```elixir
alias Explorer.DataFrame
alias Explorer.Series

# Create sample data, (or use your own data)
dates = Date.range(~D[2020-01-01], ~D[2022-12-31])
y = Enum.map(dates, fn date ->
  day_of_year = Date.day_of_year(date)
  trend = 100 + 0.1 * Date.diff(date, ~D[2020-01-01])
  seasonality = 10 * :math.sin(2 * :math.pi * day_of_year / 365.25)
  trend + seasonality + :rand.normal(0, 5)
end)

df = DataFrame.new(%{
  "ds" => dates,
  "y" => y
})

# Create and fit the model
model = Soothsayer.new()
fitted_model = Soothsayer.fit(model, df)

# Make predictions
future_dates = Date.range(~D[2023-01-01], ~D[2023-12-31])
predictions = Soothsayer.predict(fitted_model, Series.from_list(Enum.to_list(future_dates)))

# Print the predictions
IO.inspect(predictions)
```

You can also get the components of the forecast:

```elixir
components = Soothsayer.predict_components(fitted_model, Series.from_list(Enum.to_list(future_dates)))

#> %{comboned: ..., trend: ..., yearly_seasonality: ..., weekly_seasonality: ...}
```

### Customizing the Model

You can customize various aspects of the model:

```elixir
model = Soothsayer.new(%{
  trend: %{enabled: true},
  seasonality: %{
    yearly: %{enabled: true, fourier_terms: 10},
    weekly: %{enabled: false}
  },
  epochs: 200,
  learning_rate: 0.005
})
```

### Using EXLA for Faster Training

Soothsayer can use EXLA (Elixir XLA) for faster training, but it's not the default backend. To use EXLA:

1. Make sure you've added EXLA to your dependencies as shown in the Installation section.

2. Set EXLA as the default backend for Nx. Add the following to your `config/config.exs` file:

   ```elixir
   config :nx, default_backend: EXLA.Backend
   ```

   Alternatively, you can set it at runtime before using Soothsayer:

   ```elixir
   Nx.global_default_backend(EXLA.Backend)
   ```

## Differences from NeuralProphet

Soothsayer is inspired by NeuralProphet but implemented in Elixir with some key differences:

1. **Elixir Ecosystem**: Soothsayer is built using Elixir and leverages libraries like Explorer for data manipulation and Axon for neural networks.

2. **Simplified Model**: Soothsayer currently offers a more streamlined model with fewer components, focusing on trend and seasonality.

3. **Data Handling**: Soothsayer uses Explorer's DataFrame for data manipulation, which may have different performance characteristics compared to pandas.

4. **Training Process**: The training process in Soothsayer is implemented using Axon and may differ in some aspects from NeuralProphet's PyTorch implementation.

## Features Not Yet Implemented

Soothsayer is a work in progress. The following NeuralProphet features are not yet implemented:

- Auto Regression
- Lagged Regressors
- Future Regressors
- Events and Holidays
- Uncertainty Estimation
- Global and Local Models (models based on geography)
- Multiplicative Seasonality (currently only additive is supported)
- Piecewise Linear Trends (currently only simple linear trend is supported)
- Advanced Regularization Options
- Automatic Changepoint Detection
- Cross-Validation and Hyperparameter Tuning

We plan to implement some of these features in future versions of Soothsayer.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Soothsayer is released under the Apache License 2.0. See the LICENSE file for details.
