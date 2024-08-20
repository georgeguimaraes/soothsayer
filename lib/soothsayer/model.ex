defmodule Soothsayer.Model do
  defstruct [:network, :params, :config, :y_normalization]

  def new(config) do
    %__MODULE__{
      network: build_network(config),
      config: config
    }
  end

  defp build_network(config) do
    trend_input = Axon.input("trend", shape: {nil, 1})

    yearly_input =
      Axon.input("yearly", shape: {nil, seasonality_input_size(config.seasonality.yearly)})

    weekly_input =
      Axon.input("weekly", shape: {nil, seasonality_input_size(config.seasonality.weekly)})

    trend = if config.trend.enabled, do: Axon.dense(trend_input, 1), else: Axon.constant(0)

    yearly =
      if config.seasonality.yearly.enabled,
        do: Axon.dense(yearly_input, 1),
        else: Axon.constant(0)

    weekly =
      if config.seasonality.weekly.enabled,
        do: Axon.dense(weekly_input, 1),
        else: Axon.constant(0)

    Axon.add([trend, yearly, weekly])
  end

  defp seasonality_input_size(seasonality_config) do
    if seasonality_config.enabled do
      2 * Map.get(seasonality_config, :fourier_terms, 3)
    else
      1
    end
  end
end
