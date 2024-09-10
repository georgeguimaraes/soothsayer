defmodule Soothsayer do
  @moduledoc """
  The main module for the Soothsayer library, providing functions for creating, fitting, and using time series forecasting models.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Model
  alias Soothsayer.Preprocessor

  @doc """
  Creates a new Soothsayer model with the given configuration.

  ## Parameters

    * `config` - A map containing the model configuration. Defaults to an empty map.

  ## Returns

    A new `Soothsayer.Model` struct.

  ## Examples

      iex> Soothsayer.new()
      %Soothsayer.Model{config: %{trend: %{enabled: true}, seasonality: %{yearly: %{enabled: true, fourier_terms: 6}, weekly: %{enabled: true, fourier_terms: 3}}, epochs: 100, learning_rate: 0.01}, network: %Axon.Node{}, params: nil}

      iex> Soothsayer.new(%{epochs: 200, learning_rate: 0.005})
      %Soothsayer.Model{config: %{trend: %{enabled: true}, seasonality: %{yearly: %{enabled: true, fourier_terms: 6}, weekly: %{enabled: true, fourier_terms: 3}}, epochs: 200, learning_rate: 0.005}, network: %Axon.Node{}, params: nil}

  """
  @spec new(map()) :: Soothsayer.Model.t()
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

  @doc """
  Fits the Soothsayer model to the provided data.

  ## Parameters

    * `model` - A `Soothsayer.Model` struct.
    * `data` - An `Explorer.DataFrame` containing the training data.

  ## Returns

    An updated `Soothsayer.Model` struct with fitted parameters.

  ## Examples

      iex> model = Soothsayer.new()
      iex> data = Explorer.DataFrame.new(%{"ds" => [...], "y" => [...]})
      iex> fitted_model = Soothsayer.fit(model, data)
      %Soothsayer.Model{...}

  """
  @spec fit(Soothsayer.Model.t(), Explorer.DataFrame.t()) :: Soothsayer.Model.t()
  def fit(%Model{} = model, %DataFrame{} = data) do
    processed_data = Preprocessor.prepare_data(data, "y", "ds", model.config.seasonality)

    y = processed_data["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)
    {y_normalized, y_mean, y_std} = normalize(y)

    x = %{
      "trend" =>
        processed_data["ds"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1),
      "yearly" => get_seasonality_input(processed_data, :yearly),
      "weekly" => get_seasonality_input(processed_data, :weekly)
    }

    {x_normalized, x_norm} = normalize_inputs(x)

    fitted_model = Model.fit(model, x_normalized, y_normalized, model.config.epochs)

    %{
      fitted_model
      | config:
          Map.put(model.config, :normalization, %{x: x_norm, y: %{mean: y_mean, std: y_std}})
    }
  end

  @doc """
  Makes predictions using a fitted Soothsayer model.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct.
    * `x` - An `Explorer.Series` containing the dates for which to make predictions.

  ## Returns

    An `Nx.Tensor` containing the predicted values.

  ## Examples

      iex> fitted_model = Soothsayer.fit(model, training_data)
      iex> future_dates = Explorer.Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])
      iex> predictions = Soothsayer.predict(fitted_model, future_dates)
      #Nx.Tensor<
        f32[3][1]
        [
          [1.5],
          [2.3],
          [3.1]
        ]
      >

  """
  @spec predict(Soothsayer.Model.t(), Explorer.Series.t()) :: Nx.Tensor.t()
  def predict(%Model{} = model, %Series{} = x) do
    %{combined: combined} = predict_components(model, x)
    combined
  end

  @doc """
  Makes predictions and returns the individual components (trend, seasonality) using a fitted Soothsayer model.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct.
    * `x` - An `Explorer.Series` containing the dates for which to make predictions.

  ## Returns

    A map containing the predicted values for each component (trend, yearly seasonality, weekly seasonality) and the combined prediction.

  ## Examples

      iex> fitted_model = Soothsayer.fit(model, training_data)
      iex> future_dates = Explorer.Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])
      iex> predictions = Soothsayer.predict_components(fitted_model, future_dates)
      %{
        combined: #Nx.Tensor<...>,
        trend: #Nx.Tensor<...>,
        yearly_seasonality: #Nx.Tensor<...>,
        weekly_seasonality: #Nx.Tensor<...>
      }

  """
  @spec predict_components(Soothsayer.Model.t(), Explorer.Series.t()) :: %{
          combined: Nx.Tensor.t(),
          trend: Nx.Tensor.t(),
          yearly_seasonality: Nx.Tensor.t(),
          weekly_seasonality: Nx.Tensor.t()
        }
  def predict_components(%Model{} = model, %Series{} = x) do
    processed_x =
      Preprocessor.prepare_data(DataFrame.new(%{"ds" => x}), nil, "ds", model.config.seasonality)

    x_input = %{
      "trend" =>
        processed_x["ds"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1),
      "yearly" => get_seasonality_input(processed_x, :yearly),
      "weekly" => get_seasonality_input(processed_x, :weekly)
    }

    x_normalized = normalize_with_params(x_input, model.config.normalization.x)

    predictions = Model.predict(model, x_normalized)

    Map.new(predictions, fn {key, node} ->
      {key, denormalize(node, model.config.normalization.y)}
    end)
  end

  defp get_seasonality_input(data, seasonality) do
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
