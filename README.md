# Soothsayer

Soothsayer is an Elixir library for time series forecasting, inspired by Facebook's Prophet and NeuralProphet. It uses neural networks to capture trends and seasonalities in time series data.

## Current Features

- **Trend Modeling**: Captures linear and non-linear trends in time series data.
- **Seasonality Modeling**:
  - Yearly seasonality using Fourier terms
  - Weekly seasonality using Fourier terms
- **Configurable Model Architecture**: Allows customization of hidden layers in the neural network.
- **Date Decomposition**: Automatically decomposes dates into relevant features (year, day of year, day of week).
- **Flexible Input**: Works with Explorer DataFrames for easy data manipulation.
- **Normalized Inputs**: Automatically normalizes input data for better model performance.

## Planned Features

- **Changepoint Detection**: Automatically detect and adjust for trend changes.
- **Holiday Effects**: Allow incorporation of holiday effects into the model.
- **External Regressors**: Support for additional regressors that may influence the time series.
- **Uncertainty Estimation**: Provide confidence intervals for forecasts.
- **Model Diagnostics**: Tools for assessing model performance and fit.
- **Automatic Seasonality Detection**: Intelligently detect and model multiple seasonalities.
- **Cross-Validation**: Built-in cross-validation for robust model evaluation.
- **Hyperparameter Tuning**: Automated tuning of model hyperparameters.
- **Anomaly Detection**: Identify and flag anomalies in the time series.
- **Multi-step Forecasting**: Native support for multi-step ahead forecasts.
- **Plotting Functions**: Built-in functions for visualizing forecasts and components.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `soothsayer` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:soothsayer, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/soothsayer>.
