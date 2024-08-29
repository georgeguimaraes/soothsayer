defmodule Soothsayer.Model do
  defstruct [:network, :params, :config]

  def new(config) do
    %__MODULE__{
      network: build_network(config),
      config: config
    }
  end

  def build_network(config) do
    trend_input = Axon.input("trend", shape: {nil, 1})
    yearly_input = Axon.input("yearly", shape: {nil, 2 * config.seasonality.yearly.fourier_terms})
    weekly_input = Axon.input("weekly", shape: {nil, 2 * config.seasonality.weekly.fourier_terms})

    trend =
      if config.trend.enabled do
        Axon.dense(trend_input, 1, activation: :linear)
      else
        Axon.constant(0)
      end

    yearly_seasonality =
      if config.seasonality.yearly.enabled do
        Axon.dense(yearly_input, 1, activation: :linear)
      else
        Axon.constant(0)
      end

    weekly_seasonality =
      if config.seasonality.weekly.enabled do
        Axon.dense(weekly_input, 1, activation: :linear)
      else
        Axon.constant(0)
      end

    combined = Axon.add([trend, yearly_seasonality, weekly_seasonality])

    Axon.container(%{
      combined: combined,
      trend: trend,
      yearly_seasonality: yearly_seasonality,
      weekly_seasonality: weekly_seasonality
    })
  end

  def fit(model, x, y, epochs) do
    {init_fn, _predict_fn} = Axon.build(model.network)
    initial_params = init_fn.(x, %{})

    trained_params =
      model.network
      |> Axon.Loop.trainer(
        &Axon.Losses.huber(&1, &2.combined, reduction: :mean),
        Polaris.Optimizers.adam(learning_rate: model.config.learning_rate)
      )
      |> Axon.Loop.run(Stream.repeatedly(fn -> {x, y} end), initial_params,
        epochs: epochs,
        iterations: elem(Nx.shape(y), 0),
        compiler: EXLA
      )

    %{model | params: trained_params}
  end

  def predict(model, x) do
    {_init_fn, predict_fn} = Axon.build(model.network)
    predict_fn.(model.params, x)
  end
end
