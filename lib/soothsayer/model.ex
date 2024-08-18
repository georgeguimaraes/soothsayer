defmodule Soothsayer.Model do
  defstruct [
    :nn_model,
    :params,
    trend_config: %{enabled: true, hidden_sizes: [64, 32]},
    seasonality_config: %{
      yearly: %{enabled: true, fourier_terms: 6},
      weekly: %{enabled: true, fourier_terms: 3}
    }
  ]

  def new(config) do
    model = struct(__MODULE__, config)
    %{model | nn_model: build(model)}
  end

  def build(%{trend_config: trend_config, seasonality_config: seasonality_config}) do
    input = Axon.input("input", shape: {nil, nil})

    trend =
      if trend_config.enabled do
        build_trend(input, trend_config)
      else
        Axon.constant(0)
      end

    seasonality =
      if seasonality_config.yearly.enabled or seasonality_config.weekly.enabled do
        build_seasonality(input, seasonality_config)
      else
        Axon.constant(0)
      end

    output = Axon.add(trend, seasonality)

    # Ensure we always have a valid Axon model
    if trend_config.enabled or seasonality_config.yearly.enabled or
         seasonality_config.weekly.enabled do
      output
    else
      # If both trend and seasonality are disabled, create a dummy model
      Axon.dense(input, 1)
    end
  end

  defp build_trend(input, trend_config) do
    hidden_sizes = Map.get(trend_config, :hidden_sizes, [64, 32])

    Enum.reduce(hidden_sizes, input, fn size, acc ->
      acc
      |> Axon.dense(size, activation: :relu)
      |> Axon.dropout(rate: 0.1)
    end)
    |> Axon.dense(1, activation: :linear)
  end

  defp build_seasonality(input, seasonality_config) do
    yearly =
      if seasonality_config.yearly.enabled do
        build_fourier_layer(input, 365.25, seasonality_config.yearly.fourier_terms)
      else
        Axon.constant(0)
      end

    weekly =
      if seasonality_config.weekly.enabled do
        build_fourier_layer(input, 7, seasonality_config.weekly.fourier_terms)
      else
        Axon.constant(0)
      end

    Axon.add(yearly, weekly)
  end

  defp build_fourier_layer(input, period, fourier_terms) do
    fourier_input = Axon.dense(input, 2 * fourier_terms)
    Axon.dense(fourier_input, 1)
  end
end
