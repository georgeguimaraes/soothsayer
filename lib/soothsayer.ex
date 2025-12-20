defmodule Soothsayer do
  @moduledoc """
  The main module for the Soothsayer library, providing functions for creating, fitting, and using time series forecasting models.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Events
  alias Soothsayer.Model
  alias Soothsayer.Seasonality
  alias Soothsayer.Trend

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
        changepoints: 10,
        changepoints_range: 0.8,
        regularization: nil
      },
      seasonality: %{
        yearly: %{enabled: true, fourier_terms: 6},
        weekly: %{enabled: true, fourier_terms: 3}
      },
      ar: %{enabled: false, lags: 0, layers: [], regularization: nil},
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
    * `opts` - Optional keyword list:
      - `:events` - An `Explorer.DataFrame` with "event" and "ds" columns.

  ## Returns

    An updated `Soothsayer.Model` struct with fitted parameters.

  ## Examples

      iex> model = Soothsayer.new()
      iex> data = Explorer.DataFrame.new(%{"ds" => [...], "y" => [...]})
      iex> fitted_model = Soothsayer.fit(model, data)
      %Soothsayer.Model{config: %{}, network: %Axon.Node{}, params: %{}}

      iex> events_df = Explorer.DataFrame.new(%{"event" => ["sale"], "ds" => [~D[2023-01-01]]})
      iex> fitted_model = Soothsayer.fit(model, data, events: events_df)
      %Soothsayer.Model{config: %{}, network: %Axon.Node{}, params: %{}}

  """
  @spec fit(Soothsayer.Model.t(), Explorer.DataFrame.t(), keyword()) :: Soothsayer.Model.t()
  def fit(%Model{} = model, %DataFrame{} = data, opts \\ []) do
    events_df = Keyword.get(opts, :events)
    validate_training_data!(data)
    processed_data = Seasonality.add_fourier_features(data, "ds", model.config.seasonality)
    # Reorder columns to put y first
    processed_data = DataFrame.select(processed_data, ["y" | processed_data.names -- ["y"]])

    y_full = processed_data["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32})
    {y_full_normalized, y_mean, y_std} = normalize(Nx.new_axis(y_full, -1))
    y_full_normalized = Nx.flatten(y_full_normalized)

    # Handle AR: create lagged inputs and truncate data
    {y_normalized, ar_input, lags} =
      if model.config.ar.enabled and model.config.ar.lags > 0 do
        lags = model.config.ar.lags
        {ar_lagged, ar_targets} = AR.create_lagged_inputs(y_full_normalized, lags)
        {ar_targets, ar_lagged, lags}
      else
        {Nx.new_axis(y_full_normalized, -1), nil, 0}
      end

    # Build features using component modules
    dates = Series.to_list(processed_data["ds"])
    {trend_full, trend_metadata} = Trend.build_features(dates, model.config)
    seasonality = Seasonality.build_features(dates, model.config)

    # Truncate inputs if AR is enabled (remove first lags rows)
    {trend, yearly, weekly} =
      if lags > 0 do
        trend_cols = Nx.axis_size(trend_full, 1)

        {
          Nx.slice(trend_full, [lags, 0], [Nx.axis_size(trend_full, 0) - lags, trend_cols]),
          Nx.slice(seasonality.yearly, [lags, 0], [
            Nx.axis_size(seasonality.yearly, 0) - lags,
            Nx.axis_size(seasonality.yearly, 1)
          ]),
          Nx.slice(seasonality.weekly, [lags, 0], [
            Nx.axis_size(seasonality.weekly, 0) - lags,
            Nx.axis_size(seasonality.weekly, 1)
          ])
        }
      else
        {trend_full, seasonality.yearly, seasonality.weekly}
      end

    x = %{
      "trend" => trend,
      "yearly" => yearly,
      "weekly" => weekly
    }

    # Add AR input if enabled
    x = if ar_input != nil, do: Map.put(x, "ar", ar_input), else: x

    # Add events input if configured
    events_config = model.config[:events] || %{}
    dates = Series.to_list(processed_data["ds"])

    # Truncate dates for events if AR is enabled
    event_dates =
      if lags > 0 do
        Enum.drop(dates, lags)
      else
        dates
      end

    x =
      if map_size(events_config) > 0 and events_df != nil do
        events_input =
          Events.build_features(Series.from_list(event_dates), events_df, events_config)

        Map.put(x, "events", events_input)
      else
        x
      end

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
          |> Map.put(:first_date, trend_metadata.first_date)
          |> Map.put(:changepoint_positions, trend_metadata.changepoint_positions)
    }
  end

  @doc """
  Makes predictions using a fitted Soothsayer model.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct.
    * `x` - An `Explorer.Series` containing the dates for which to make predictions.
    * `opts` - Optional keyword list:
      - `:events` - An `Explorer.DataFrame` with "event" and "ds" columns.

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
  @spec predict(Soothsayer.Model.t(), Explorer.Series.t(), keyword()) :: Nx.Tensor.t()
  def predict(%Model{} = model, %Series{} = x, opts \\ []) do
    %{combined: combined} = predict_components(model, x, opts)
    combined
  end

  @doc """
  Makes predictions and returns the individual components (trend, seasonality) using a fitted Soothsayer model.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct.
    * `x` - An `Explorer.Series` containing the dates for which to make predictions.
    * `opts` - Optional keyword list:
      - `:events` - An `Explorer.DataFrame` with "event" and "ds" columns.

  ## Returns

    A map containing the predicted values for each component (trend, yearly seasonality, weekly seasonality, events) and the combined prediction.

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
  @spec predict_components(Soothsayer.Model.t(), Explorer.Series.t(), keyword()) :: %{
          combined: Nx.Tensor.t(),
          trend: Nx.Tensor.t(),
          yearly_seasonality: Nx.Tensor.t(),
          weekly_seasonality: Nx.Tensor.t()
        }
  def predict_components(%Model{} = model, %Series{} = x, opts \\ []) do
    events_df = Keyword.get(opts, :events)

    prediction_dates = Series.to_list(x)

    # Build trend input using stored changepoint positions from training
    first_date = model.config.first_date
    changepoint_positions = model.config.changepoint_positions
    t = Trend.date_to_numeric(prediction_dates, first_date) |> Nx.new_axis(-1)
    changepoint_features = Trend.build_changepoint_features(t, changepoint_positions)
    trend_input = Trend.build_trend_input(t, changepoint_features)

    # Build seasonality features
    seasonality = Seasonality.build_features(prediction_dates, model.config)

    x_input = %{
      "trend" => trend_input,
      "yearly" => seasonality.yearly,
      "weekly" => seasonality.weekly
    }

    # Add AR input if enabled
    x_input =
      if model.config.ar.enabled and model.config.ar.lags > 0 do
        ar_input =
          AR.build_input(model.config.training_data, Series.to_list(x), model.config.ar.lags)

        Map.put(x_input, "ar", ar_input)
      else
        x_input
      end

    # Add events input if configured
    events_config = model.config[:events] || %{}

    x_input =
      if map_size(events_config) > 0 and events_df != nil do
        events_input = Events.build_features(x, events_df, events_config)
        Map.put(x_input, "events", events_input)
      else
        x_input
      end

    x_normalized = normalize_with_params(x_input, model.config.normalization.x)

    predictions = Model.predict(model, x_normalized)

    Map.new(predictions, fn {key, node} ->
      {key, denormalize(node, model.config.normalization.y)}
    end)
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

      iex> model = Soothsayer.new(%{ar: %{enabled: true, lags: 3}})
      iex> fitted_model = Soothsayer.fit(model, data)
      iex> weights = Soothsayer.get_ar_weights(fitted_model)
      %{
        "ar_dense_out" => %{kernel: #Nx.Tensor<f32[3][1]>, bias: #Nx.Tensor<f32[1]>}
      }

  """
  @spec get_ar_weights(Soothsayer.Model.t()) :: %{
          String.t() => %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}
        }
  def get_ar_weights(%Model{} = model) do
    AR.get_weights(model)
  end

  @doc """
  Extracts the learned event coefficients from a fitted model.

  Returns a map of event feature names to their learned coefficients.
  Feature names are formatted as "event_name_offset" where offset indicates
  the window position relative to the event date.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct with events configured.

  ## Returns

    A map of feature names to coefficient values.

  ## Examples

      iex> model = Soothsayer.new(%{events: %{"sale" => %{lower_window: 0, upper_window: 0}}})
      iex> fitted_model = Soothsayer.fit(model, data, events: events_df)
      iex> effects = Soothsayer.get_event_effects(fitted_model)
      %{"sale_0" => 45.2}

      iex> model = Soothsayer.new(%{events: %{"promo" => %{lower_window: -1, upper_window: 1}}})
      iex> fitted_model = Soothsayer.fit(model, data, events: events_df)
      iex> effects = Soothsayer.get_event_effects(fitted_model)
      %{"promo_-1" => 12.5, "promo_0" => 50.0, "promo_+1" => 8.3}

  """
  @spec get_event_effects(Soothsayer.Model.t()) :: %{String.t() => float()}
  def get_event_effects(%Model{} = model) do
    Events.get_effects(model)
  end

  @doc """
  Returns a display-friendly version of the network that outputs a single tensor.

  This can be used with `Axon.Display.as_graph/2` since it doesn't use
  `Axon.container` with a map output.

  ## Parameters

    * `model` - A `Soothsayer.Model` struct.

  ## Returns

    An Axon network suitable for visualization.

  ## Examples

      iex> model = Soothsayer.new()
      iex> input = %{
      ...>   "trend" => Nx.template({1, 11}, :f32),
      ...>   "yearly" => Nx.template({1, 12}, :f32),
      ...>   "weekly" => Nx.template({1, 6}, :f32)
      ...> }
      iex> Axon.Display.as_graph(Soothsayer.display_network(model), input)

  """
  @spec display_network(Soothsayer.Model.t()) :: Axon.t()
  def display_network(%Model{} = model) do
    Model.display_network(model.config)
  end
end
