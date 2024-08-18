defmodule Soothsayer.Model do
  defstruct [
    :trend_model,
    :yearly_seasonality_model,
    :weekly_seasonality_model,
    :params,
    :epochs,
    :y_normalization,
    trend_config: %{enabled: true},
    seasonality_config: %{
      yearly: %{enabled: true, fourier_terms: 6},
      weekly: %{enabled: true, fourier_terms: 3}
    }
  ]

  def new(config) do
    model = struct(__MODULE__, config)

    %{
      model
      | trend_model: build_trend(model.trend_config),
        yearly_seasonality_model: build_seasonality(model.seasonality_config.yearly, :yearly),
        weekly_seasonality_model: build_seasonality(model.seasonality_config.weekly, :weekly)
    }
  end

  defp build_trend(%{enabled: true}) do
    input = Axon.input("trend_input", shape: {nil, 1})
    Axon.dense(input, 1, activation: :linear)
  end

  defp build_trend(%{enabled: false}), do: nil

  defp build_seasonality(%{enabled: true, fourier_terms: terms}, name) do
    input = Axon.input("#{name}_input", shape: {nil, 2 * terms})
    Axon.dense(input, 1, activation: :relu)
  end

  defp build_seasonality(%{enabled: false}, _name), do: nil

  def fit_model(model, x, y, epochs) when not is_nil(model) do
    {init_fn, predict_fn} = Axon.build(model)
    initial_params = init_fn.(x, %{})

    trained_params =
      model
      |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adam(learning_rate: 0.01))
      |> Axon.Loop.run(Stream.repeatedly(fn -> {x, y} end), initial_params,
        epochs: epochs,
        iterations: elem(Nx.shape(y), 0),
        compiler: EXLA
      )

    {predict_fn, trained_params}
  end

  def fit_model(nil, _x, _y, _epochs), do: {nil, nil}

  def predict(model, params, x) when not is_nil(model) do
    {_init_fn, predict_fn} = Axon.build(model)
    predict_fn.(params, x)
  end

  def predict(nil, _params, x), do: Nx.broadcast(0, {elem(Nx.shape(x), 0), 1})
end
