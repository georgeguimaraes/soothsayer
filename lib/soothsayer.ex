defmodule Soothsayer do
  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Model
  alias Soothsayer.Preprocessor

  def new(config \\ %{}) do
    default_config = %{
      trend: %{enabled: true},
      seasonality: %{
        yearly: %{enabled: true, fourier_terms: 6},
        weekly: %{enabled: true, fourier_terms: 3}
      },
      epochs: 100,
      learning_rate: 0.001
    }

    merged_config = deep_merge(default_config, config)
    Model.new(merged_config)
  end

  defp deep_merge(left, right) do
    Map.merge(left, right, fn
      _, %{} = left, %{} = right -> deep_merge(left, right)
      _, _left, right -> right
    end)
  end

  def fit(%Model{} = model, %DataFrame{} = data) do
    processed_data = Preprocessor.prepare_data(data, "y", "ds", model.config.seasonality)

    y = processed_data["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)
    {y_normalized, y_mean, y_std} = normalize(y)

    x = prepare_inputs(processed_data, model.config)

    {init_fn, predict_fn} = Axon.build(model.network)
    initial_params = init_fn.(x, %{})

    trained_params =
      model.network
      |> Axon.Loop.trainer(
        :mean_squared_error,
        Polaris.Optimizers.adam(learning_rate: model.config.learning_rate)
      )
      |> Axon.Loop.run(Stream.repeatedly(fn -> {x, y_normalized} end), initial_params,
        epochs: model.config.epochs,
        iterations: elem(Nx.shape(y), 0),
        compiler: EXLA
      )

    %{model | params: trained_params, y_normalization: %{mean: y_mean, std: y_std}}
  end

  def predict(%Model{} = model, %Series{} = x) do
    processed_x =
      Preprocessor.prepare_data(DataFrame.new(%{"ds" => x}), nil, "ds", model.config.seasonality)

    x_prepared = prepare_inputs(processed_x, model.config)

    {_, predict_fn} = Axon.build(model.network)
    predictions = predict_fn.(model.params, x_prepared)

    denormalize(predictions, model.y_normalization) |> Nx.to_flat_list()
  end

  defp prepare_inputs(data, config) do
    trend_input = data["ds"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)
    yearly_input = get_seasonality_input(data, :yearly, config.seasonality.yearly)
    weekly_input = get_seasonality_input(data, :weekly, config.seasonality.weekly)

    %{
      "trend" => trend_input,
      "yearly" => yearly_input,
      "weekly" => weekly_input
    }
  end

  defp get_seasonality_input(data, seasonality, seasonality_config) do
    if seasonality_config.enabled do
      fourier_terms = Map.get(seasonality_config, :fourier_terms, 3)
      columns = data.names |> Enum.filter(&String.starts_with?(&1, Atom.to_string(seasonality)))

      data[columns]
      |> DataFrame.to_series()
      |> Map.values()
      |> Enum.map(&Series.to_tensor/1)
      |> Nx.stack(axis: 1)
      |> Nx.as_type({:f, 32})
    else
      Nx.tensor([[0.0]])
    end
  end

  defp normalize(tensor) do
    mean = Nx.mean(tensor, axes: [0])
    std = Nx.standard_deviation(tensor, axes: [0])
    std = Nx.select(Nx.equal(std, 0), Nx.tensor(1), std)
    {Nx.divide(Nx.subtract(tensor, mean), std), mean, std}
  end

  defp denormalize(tensor, %{mean: mean, std: std}) do
    Nx.multiply(tensor, std) |> Nx.add(mean)
  end
end
