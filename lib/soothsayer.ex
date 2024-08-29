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
      learning_rate: 0.01
    }

    merged_config = deep_merge(default_config, config)
    Model.new(merged_config)
  end

  def fit(%Model{} = model, %DataFrame{} = data) do
    processed_data = Preprocessor.prepare_data(data, "y", "ds", model.config.seasonality)

    y = processed_data["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)
    {y_normalized, y_mean, y_std} = normalize(y)

    x = %{
      "trend" =>
        processed_data["ds"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1),
      "yearly" => get_seasonality_input(processed_data, :yearly, model.config),
      "weekly" => get_seasonality_input(processed_data, :weekly, model.config)
    }

    {x_normalized, x_norm} = normalize_inputs(x)

    fitted_model = Model.fit(model, x_normalized, y_normalized, model.config.epochs)

    %{
      fitted_model
      | config:
          Map.put(model.config, :normalization, %{x: x_norm, y: %{mean: y_mean, std: y_std}})
    }
  end

  def predict(%Model{} = model, %Series{} = x) do
    %{combined: combined} = predict_components(model, x)
    combined
  end

  def predict_components(%Model{} = model, %Series{} = x) do
    processed_x =
      Preprocessor.prepare_data(DataFrame.new(%{"ds" => x}), nil, "ds", model.config.seasonality)

    x_input = %{
      "trend" =>
        processed_x["ds"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1),
      "yearly" => get_seasonality_input(processed_x, :yearly, model.config),
      "weekly" => get_seasonality_input(processed_x, :weekly, model.config)
    }

    x_normalized = normalize_with_params(x_input, model.config.normalization.x)

    predictions = Model.predict(model, x_normalized)

    Map.new(predictions, fn {key, node} ->
      {key, denormalize(node, model.config.normalization.y)}
    end)
  end

  defp get_seasonality_input(data, seasonality, config) do
    columns = data.names |> Enum.filter(&String.starts_with?(&1, Atom.to_string(seasonality)))

    data[columns]
    |> DataFrame.to_series()
    |> Map.values()
    |> Enum.map(&Series.to_tensor/1)
    |> Nx.stack(axis: 1)
    |> Nx.as_type({:f, 32})
  end

  defp normalize(tensor) do
    mean = Nx.mean(tensor, axes: [0])
    std = Nx.standard_deviation(tensor, axes: [0])
    std = Nx.select(Nx.equal(std, 0), Nx.tensor(1), std)
    {Nx.divide(Nx.subtract(tensor, mean), std), mean, std}
  end

  defp normalize_inputs(x) do
    Enum.reduce(x, {%{}, %{}}, fn {key, tensor}, {normalized, norm_params} ->
      {normalized_tensor, mean, std} = normalize(tensor)

      {Map.put(normalized, key, normalized_tensor),
       Map.put(norm_params, key, %{mean: mean, std: std})}
    end)
  end

  defp normalize_with_params(x, norm_params) do
    Enum.map(x, fn {key, tensor} ->
      mean = norm_params[key].mean
      std = norm_params[key].std
      {key, Nx.divide(Nx.subtract(tensor, mean), std)}
    end)
    |> Enum.into(%{})
  end

  defp denormalize(tensor, %{mean: mean, std: std}) do
    Nx.add(Nx.multiply(tensor, std), mean)
  end

  defp deep_merge(left, right) do
    Map.merge(left, right, fn
      _, %{} = left, %{} = right -> deep_merge(left, right)
      _, _left, right -> right
    end)
  end
end
