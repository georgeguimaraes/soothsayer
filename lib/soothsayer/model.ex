defmodule Soothsayer.Model do
  defstruct [:nn_model, :params, :trend_config]

  def build(config) do
    input = Axon.input("input", shape: {nil, 1})
    trend = build_trend(input, config.trend_config)
    seasonality = build_seasonality(input, config.seasonality_config)

    trend
    |> Axon.add(seasonality)
  end

  def new(config \\ %{}) do
    trend_config = Map.get(config, :trend, %{hidden_sizes: [64, 32]})
    seasonality_config = Map.get(config, :seasonality, %{})

    nn_model = build(%{trend_config: trend_config, seasonality_config: seasonality_config})
    %__MODULE__{nn_model: nn_model, params: nil, trend_config: trend_config}
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

  defp build_seasonality(_input, _config) do
    # Placeholder for seasonality component
    # We'll implement this in the next step
    Axon.constant(0)
  end
end
