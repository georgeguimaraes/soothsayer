defmodule Soothsayer do
  @moduledoc """
  The main module for the Soothsayer library, providing functions for creating, fitting, and using time series forecasting models.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Changepoints
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
      trend: %{
        enabled: true,
        n_changepoints: 10,
        changepoints_range: 0.8,
        regularization: nil
      },
      seasonality: %{
        yearly: %{enabled: true, fourier_terms: 6},
        weekly: %{enabled: true, fourier_terms: 3}
      },
      ar: %{enabled: false, n_lags: 0, layers: [], regularization: nil},
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
      %Soothsayer.Model{config: %{}, network: %Axon.Node{}, params: %{}}

  """
  @spec fit(Soothsayer.Model.t(), Explorer.DataFrame.t()) :: Soothsayer.Model.t()
  def fit(%Model{} = model, %DataFrame{} = data) do
    validate_training_data!(data)
    processed_data = Preprocessor.prepare_data(data, "y", "ds", model.config.seasonality)

    y_full = processed_data["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32})
    {y_full_normalized, y_mean, y_std} = normalize(Nx.new_axis(y_full, -1))
    y_full_normalized = Nx.flatten(y_full_normalized)

    # Handle AR: create lagged inputs and truncate data
    {y_normalized, ar_input, n_lags} =
      if model.config.ar.enabled and model.config.ar.n_lags > 0 do
        n_lags = model.config.ar.n_lags
        {ar_lagged, ar_targets} = AR.create_lagged_inputs(y_full_normalized, n_lags)
        {ar_targets, ar_lagged, n_lags}
      else
        {Nx.new_axis(y_full_normalized, -1), nil, 0}
      end

    # Extract dates and compute changepoint info
    dates = Series.to_list(processed_data["ds"])
    first_date = List.first(dates)
    n_changepoints = model.config.trend.n_changepoints
    changepoints_range = model.config.trend.changepoints_range

    # Compute changepoint positions as numeric values (days since first date)
    changepoint_positions = compute_numeric_changepoint_positions(dates, first_date, n_changepoints, changepoints_range)

    # Build base time values tensor (days since first date)
    t_full = Changepoints.date_to_numeric(dates, first_date) |> Nx.new_axis(-1)

    # Build changepoint features
    changepoint_features_full = Changepoints.build_changepoint_features(t_full, changepoint_positions)

    # Build trend input with changepoint features
    trend_full = Changepoints.build_trend_input(t_full, changepoint_features_full)

    yearly_full =
      get_seasonality_input(
        processed_data,
        :yearly,
        model.config.seasonality.yearly.fourier_terms
      )

    weekly_full =
      get_seasonality_input(
        processed_data,
        :weekly,
        model.config.seasonality.weekly.fourier_terms
      )

    # Truncate inputs if AR is enabled (remove first n_lags rows)
    {trend, yearly, weekly} =
      if n_lags > 0 do
        trend_cols = Nx.axis_size(trend_full, 1)
        {
          Nx.slice(trend_full, [n_lags, 0], [Nx.axis_size(trend_full, 0) - n_lags, trend_cols]),
          Nx.slice(yearly_full, [n_lags, 0], [Nx.axis_size(yearly_full, 0) - n_lags, Nx.axis_size(yearly_full, 1)]),
          Nx.slice(weekly_full, [n_lags, 0], [Nx.axis_size(weekly_full, 0) - n_lags, Nx.axis_size(weekly_full, 1)])
        }
      else
        {trend_full, yearly_full, weekly_full}
      end

    x = %{
      "trend" => trend,
      "yearly" => yearly,
      "weekly" => weekly
    }

    # Add AR input if enabled
    x = if ar_input != nil, do: Map.put(x, "ar", ar_input), else: x

    {x_normalized, x_norm} = normalize_inputs(x)

    fitted_model = Model.fit(model, x_normalized, y_normalized, model.config.epochs)

    # Store training data for prediction lookups
    training_data = %{
      dates: dates,
      y_normalized: Nx.to_flat_list(y_full_normalized)
    }

    %{
      fitted_model
      | config:
          model.config
          |> Map.put(:normalization, %{x: x_norm, y: %{mean: y_mean, std: y_std}})
          |> Map.put(:training_data, training_data)
          |> Map.put(:first_date, first_date)
          |> Map.put(:changepoint_positions, changepoint_positions)
    }
  end

  defp compute_numeric_changepoint_positions(dates, first_date, n_changepoints, changepoints_range) do
    changepoint_dates = Changepoints.compute_changepoint_positions(dates, n_changepoints, changepoints_range)
    Enum.map(changepoint_dates, fn date -> Date.diff(date, first_date) * 1.0 end)
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

    # Build trend input with changepoint features using stored positions
    prediction_dates = Series.to_list(x)
    first_date = model.config.first_date
    changepoint_positions = model.config.changepoint_positions

    t = Changepoints.date_to_numeric(prediction_dates, first_date) |> Nx.new_axis(-1)
    changepoint_features = Changepoints.build_changepoint_features(t, changepoint_positions)
    trend_input = Changepoints.build_trend_input(t, changepoint_features)

    x_input = %{
      "trend" => trend_input,
      "yearly" =>
        get_seasonality_input(processed_x, :yearly, model.config.seasonality.yearly.fourier_terms),
      "weekly" =>
        get_seasonality_input(processed_x, :weekly, model.config.seasonality.weekly.fourier_terms)
    }

    # Add AR input if enabled
    x_input =
      if model.config.ar.enabled and model.config.ar.n_lags > 0 do
        ar_input = AR.build_input(model.config.training_data, Series.to_list(x), model.config.ar.n_lags)
        Map.put(x_input, "ar", ar_input)
      else
        x_input
      end

    x_normalized = normalize_with_params(x_input, model.config.normalization.x)

    predictions = Model.predict(model, x_normalized)

    Map.new(predictions, fn {key, node} ->
      {key, denormalize(node, model.config.normalization.y)}
    end)
  end

  defp get_seasonality_input(data, seasonality, fourier_terms) do
    columns = data.names |> Enum.filter(&String.starts_with?(&1, Atom.to_string(seasonality)))

    case columns do
      [] ->
        # No seasonality columns found - return zero tensor with expected shape
        row_count = DataFrame.n_rows(data)
        col_count = 2 * fourier_terms
        Nx.broadcast(0.0, {row_count, col_count}) |> Nx.as_type({:f, 32})

      _ ->
        data[columns]
        |> DataFrame.to_series()
        |> Map.values()
        |> Enum.map(&Series.to_tensor/1)
        |> Nx.stack(axis: 1)
        |> Nx.as_type({:f, 32})
    end
  end

  defp normalize(tensor) do
    mean = Nx.mean(tensor, axes: [0])
    std = Nx.standard_deviation(tensor, axes: [0])
    std = Nx.select(Nx.equal(std, 0), Nx.tensor(1), std)
    {Nx.divide(Nx.subtract(tensor, mean), std), mean, std}
  end

  defp normalize_inputs(x) do
    Enum.reduce(x, {%{}, %{}}, fn {key, tensor}, acc ->
      normalize_single_input(key, tensor, acc)
    end)
  end

  defp normalize_single_input(key, tensor, {normalized, norm_params}) do
    {normalized_tensor, mean, std} = normalize(tensor)
    norm_param = %{mean: mean, std: std}

    {Map.put(normalized, key, normalized_tensor), Map.put(norm_params, key, norm_param)}
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

  defp validate_training_data!(%DataFrame{} = data) do
    row_count = DataFrame.n_rows(data)
    columns = DataFrame.names(data)

    cond do
      row_count < 2 ->
        raise ArgumentError, "Training data must have at least 2 rows, got #{row_count}"

      "ds" not in columns ->
        raise ArgumentError,
              "Training data must contain a 'ds' (date) column. Available columns: #{inspect(columns)}"

      "y" not in columns ->
        raise ArgumentError,
              "Training data must contain a 'y' (target values) column. Available columns: #{inspect(columns)}"

      true ->
        :ok
    end
  end

  defp deep_merge(left, right) do
    Map.merge(left, right, fn
      _, %{} = left, %{} = right -> deep_merge(left, right)
      _, _left, right -> right
    end)
  end

  @doc """
  Extracts the raw AR layer weights from a fitted model.

  For linear AR models, returns the output layer weights.
  For deep AR-Net models, returns all layer weights including hidden layers.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct with AR enabled.

  ## Returns

    A map of layer names to weight structs containing `:kernel` and `:bias` tensors.

  ## Examples

      iex> model = Soothsayer.new(%{ar: %{enabled: true, n_lags: 3}})
      iex> fitted_model = Soothsayer.fit(model, data)
      iex> weights = Soothsayer.get_ar_weights(fitted_model)
      %{
        "ar_dense_out" => %{kernel: #Nx.Tensor<f32[3][1]>, bias: #Nx.Tensor<f32[1]>}
      }

  """
  @spec get_ar_weights(Soothsayer.Model.t()) :: %{String.t() => %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}}
  def get_ar_weights(%Model{} = model) do
    AR.get_weights(model)
  end
end
