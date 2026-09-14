defmodule Soothsayer do
  @moduledoc """
  The main module for the Soothsayer library, providing functions for creating, fitting, and using time series forecasting models.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Events
  alias Soothsayer.Model
  alias Soothsayer.Regressors
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
        mode: :additive,
        yearly: %{enabled: true, fourier_terms: 6},
        weekly: %{enabled: true, fourier_terms: 3}
      },
      ar: %{enabled: false, lags: 0, layers: [], regularization: nil},
      regressors: [],
      epochs: 100,
      learning_rate: 0.01,
      batch_size: nil,
      seed: nil
    }

    merged_config = deep_merge(default_config, config)
    validate_config!(merged_config)
    Model.new(merged_config)
  end

  @seasonality_modes [:additive, :multiplicative]

  defp validate_config!(%{seasonality: %{mode: mode}}) when mode not in @seasonality_modes do
    raise ArgumentError,
          "seasonality.mode must be one of #{inspect(@seasonality_modes)}, got #{inspect(mode)}"
  end

  defp validate_config!(%{regressors: regressors}) when not is_list(regressors) do
    raise ArgumentError,
          "regressors must be a list of column names, got #{inspect(regressors)}"
  end

  defp validate_config!(%{regressors: regressors}) do
    for name <- regressors, not is_binary(name) do
      raise ArgumentError, "regressors must be column name strings, got #{inspect(name)}"
    end

    :ok
  end

  @doc """
  Fits the Soothsayer model to the provided data.

  ## Parameters

    * `model` - A `Soothsayer.Model` struct.
    * `data` - An `Explorer.DataFrame` containing the training data.
    * `opts` - Optional keyword list:
      - `:events` - An `Explorer.DataFrame` with "event" and "ds" columns.

    When the model config lists `regressors`, `data` must contain a column
    for each of them.

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
    Regressors.validate_columns!(data, model.config.regressors)
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

    # Events and regressors line up with the (possibly AR-truncated) targets
    feature_dates = Enum.drop(dates, lags)

    x =
      x
      |> put_events_input(model, feature_dates, events_df)
      |> put_regressors_input(model, feature_dates, data)

    {x_normalized, x_norm} = normalize_inputs(x)

    # The network is rebuilt now that the y normalization is known, since
    # multiplicative seasonality needs the series level as a constant.
    config =
      Map.put(model.config, :normalization, %{x: x_norm, y: %{mean: y_mean, std: y_std}})

    model = %{model | config: config, network: Model.build_network(config)}

    fitted_model = Model.fit(model, x_normalized, y_normalized, config.epochs)

    # Store training data for prediction lookups
    training_data = %{
      dates: dates,
      y_normalized: Nx.to_flat_list(y_full_normalized)
    }

    %{
      fitted_model
      | config:
          config
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
      - `:history` - An `Explorer.DataFrame` with "ds" and "y" columns holding
        observations newer than the training data. Only used when
        auto-regression is enabled, see `predict_components/3`.
      - `:regressors` - An `Explorer.DataFrame` with "ds" plus one column per
        configured regressor, covering every predicted date. Required when
        the model was fitted with regressors.

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
      - `:history` - An `Explorer.DataFrame` with "ds" and "y" columns holding
        observations newer than the training data. Only used when
        auto-regression is enabled.
      - `:regressors` - An `Explorer.DataFrame` with "ds" plus one column per
        configured regressor. Required when the model was fitted with
        regressors, and it must cover every predicted date. With
        auto-regression it must also cover the days between the last
        observation and the forecast, since those get predicted too.

  ## Auto-regression and future dates

    When AR is enabled, each prediction needs the `lags` values before it.
    Dates inside the training data (or `:history`) use the observed values.
    Dates past the last observation are forecast one day at a time from the
    last observation forward, feeding each prediction back in as the next
    day's lag, up to the latest requested date. Errors compound over that
    horizon, so far-out AR forecasts revert toward the level the model
    learned. This assumes daily, gap-free data.

  ## Returns

    A map with the combined prediction and each component: `:trend`,
    `:yearly_seasonality`, `:weekly_seasonality`, `:ar`, `:events` and
    `:regressors`.

    The components add up to `:combined`. Trend carries the level of the
    series, so the other components are zero-centered offsets around it.
    Components that are disabled in the config are all zeros. When trend is
    disabled, `:trend` is a flat line at the training mean.

    With `seasonality: %{mode: :multiplicative}` the seasonal components are
    still returned in absolute units (the amount added to the trend on that
    date), not as fractions of the trend, so they keep summing to `:combined`.

  ## Examples

      iex> fitted_model = Soothsayer.fit(model, training_data)
      iex> future_dates = Explorer.Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])
      iex> predictions = Soothsayer.predict_components(fitted_model, future_dates)
      %{
        combined: #Nx.Tensor<...>,
        trend: #Nx.Tensor<...>,
        yearly_seasonality: #Nx.Tensor<...>,
        weekly_seasonality: #Nx.Tensor<...>,
        ar: #Nx.Tensor<...>,
        events: #Nx.Tensor<...>,
        regressors: #Nx.Tensor<...>
      }

  """
  @spec predict_components(Soothsayer.Model.t(), Explorer.Series.t(), keyword()) :: %{
          combined: Nx.Tensor.t(),
          trend: Nx.Tensor.t(),
          yearly_seasonality: Nx.Tensor.t(),
          weekly_seasonality: Nx.Tensor.t(),
          ar: Nx.Tensor.t(),
          events: Nx.Tensor.t(),
          regressors: Nx.Tensor.t()
        }
  def predict_components(%Model{} = model, %Series{} = x, opts \\ []) do
    events_df = Keyword.get(opts, :events)
    history = Keyword.get(opts, :history)
    regressors_df = Keyword.get(opts, :regressors)
    validate_regressors_option!(model, regressors_df)

    prediction_dates = Series.to_list(x)
    x_input = build_time_inputs(model, prediction_dates, events_df, regressors_df)

    x_input =
      if ar_enabled?(model) do
        known_values =
          model
          |> known_values(history)
          |> forecast_missing_values(model, prediction_dates, events_df, regressors_df)

        ar_input = AR.build_input(known_values, prediction_dates, model.config.ar.lags)
        Map.put(x_input, "ar", ar_input)
      else
        x_input
      end

    x_normalized = normalize_with_params(x_input, model.config.normalization.x)

    predictions = Model.predict(model, x_normalized)

    denormalize_components(predictions, model.config.normalization.y)
  end

  defp put_events_input(x, model, dates, events_df) do
    events_config = model.config[:events] || %{}

    if map_size(events_config) > 0 and events_df != nil do
      events_input = Events.build_features(Series.from_list(dates), events_df, events_config)
      Map.put(x, "events", events_input)
    else
      x
    end
  end

  defp put_regressors_input(x, model, dates, regressors_df) do
    case model.config.regressors do
      [] -> x
      names -> Map.put(x, "regressors", Regressors.build_features(dates, regressors_df, names))
    end
  end

  defp ar_enabled?(model), do: model.config.ar.enabled and model.config.ar.lags > 0

  defp validate_regressors_option!(%Model{config: %{regressors: []}}, _regressors_df), do: :ok

  defp validate_regressors_option!(%Model{config: %{regressors: names}}, nil) do
    raise ArgumentError,
          "This model was fitted with regressors #{inspect(names)}. " <>
            "Pass regressors: a dataframe with \"ds\" and those columns to predict."
  end

  defp validate_regressors_option!(_model, %DataFrame{}), do: :ok

  # Builds every input that depends only on the date: trend, seasonality,
  # events and regressors. AR is added separately since it depends on
  # previous values.
  defp build_time_inputs(model, dates, events_df, regressors_df) do
    t = Trend.date_to_numeric(dates, model.config.first_date) |> Nx.new_axis(-1)
    changepoint_features = Trend.build_changepoint_features(t, model.config.changepoint_positions)
    trend_input = Trend.build_trend_input(t, changepoint_features)

    seasonality = Seasonality.build_features(dates, model.config)

    %{
      "trend" => trend_input,
      "yearly" => seasonality.yearly,
      "weekly" => seasonality.weekly
    }
    |> put_events_input(model, dates, events_df)
    |> put_regressors_input(model, dates, regressors_df)
  end

  # Observed values in normalized y space, keyed by date. Training data comes
  # first, then `history` overrides or extends it.
  defp known_values(model, nil), do: AR.known_values(model.config.training_data)

  defp known_values(model, %DataFrame{} = history) do
    validate_history!(history)
    %{mean: mean, std: std} = model.config.normalization.y

    history_values =
      history["y"]
      |> Series.to_tensor()
      |> Nx.as_type({:f, 32})
      |> Nx.subtract(mean)
      |> Nx.divide(std)
      |> Nx.to_flat_list()

    history_dates = Series.to_list(history["ds"])

    Map.merge(known_values(model, nil), Map.new(Enum.zip(history_dates, history_values)))
  end

  # Walks day by day from the last known date to the latest prediction date,
  # predicting each day from the days before it and recording the prediction
  # as that day's known value. Returns the extended map. Nothing is stored on
  # the model.
  defp forecast_missing_values(known_values, model, prediction_dates, events_df, regressors_df) do
    last_known_date = known_values |> Map.keys() |> Enum.max(Date)
    last_prediction_date = Enum.max(prediction_dates, Date)

    if Date.compare(last_prediction_date, last_known_date) == :gt do
      rollout_dates = Date.range(Date.add(last_known_date, 1), last_prediction_date)
      roll_forward(known_values, model, Enum.to_list(rollout_dates), events_df, regressors_df)
    else
      known_values
    end
  end

  defp roll_forward(known_values, model, rollout_dates, events_df, regressors_df) do
    time_inputs = build_time_inputs(model, rollout_dates, events_df, regressors_df)

    rollout_dates
    |> Enum.with_index()
    |> Enum.reduce(known_values, fn {date, row}, known_values ->
      prediction = forecast_one_day(model, known_values, date, slice_row(time_inputs, row))
      Map.put(known_values, date, prediction)
    end)
  end

  defp slice_row(inputs, row) do
    Map.new(inputs, fn {key, tensor} -> {key, Nx.slice_along_axis(tensor, row, 1, axis: 0)} end)
  end

  # Returns the day's combined prediction in normalized y space, ready to be
  # used as a lag for the following day.
  defp forecast_one_day(model, known_values, date, time_inputs) do
    inputs =
      time_inputs
      |> Map.put("ar", AR.build_input(known_values, [date], model.config.ar.lags))
      |> normalize_with_params(model.config.normalization.x)

    %{combined: combined} = Model.predict(model, inputs)
    combined |> Nx.reshape({}) |> Nx.to_number()
  end

  # The network predicts in normalized y space, where every component is a
  # zero-centered offset and the series mean lives outside the network. When
  # denormalizing, only `combined` and `trend` get the mean added back, so
  # trend owns the level and the components sum to `combined`. Disabled
  # components stay at zero instead of collapsing to the series mean.
  defp denormalize_components(predictions, %{mean: mean, std: std}) do
    Map.new(predictions, fn
      {key, tensor} when key in [:combined, :trend] ->
        {key, Nx.add(Nx.multiply(tensor, std), mean)}

      {key, tensor} ->
        {key, Nx.multiply(tensor, std)}
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

  defp validate_history!(%DataFrame{} = history) do
    columns = DataFrame.names(history)

    for required <- ["ds", "y"], required not in columns do
      raise ArgumentError,
            "History must contain a '#{required}' column. Available columns: #{inspect(columns)}"
    end

    :ok
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
  Extracts the learned future regressor coefficients from a fitted model.

  Coefficients are in normalized units: the change in normalized y for a one
  standard deviation change in the regressor. Positive means the regressor
  pushes the forecast up.

  ## Examples

      iex> model = Soothsayer.new(%{regressors: ["temperature"]})
      iex> fitted_model = Soothsayer.fit(model, data)
      iex> Soothsayer.get_regressor_effects(fitted_model)
      %{"temperature" => 0.42}

  """
  @spec get_regressor_effects(Soothsayer.Model.t()) :: %{String.t() => float()}
  def get_regressor_effects(%Model{} = model) do
    Regressors.get_effects(model)
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
