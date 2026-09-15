defmodule Soothsayer do
  @moduledoc """
  The main module for the Soothsayer library, providing functions for creating, fitting, and using time series forecasting models.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Events
  alias Soothsayer.LaggedRegressors
  alias Soothsayer.Model
  alias Soothsayer.Quantiles
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
      ar: %{enabled: false, lags: 0, layers: [], regularization: nil, forecast_steps: 1},
      regressors: [],
      lagged_regressors: %{},
      quantiles: [],
      epochs: 100,
      learning_rate: 0.01,
      batch_size: nil,
      seed: nil
    }

    merged_config = deep_merge(default_config, config)
    validate_config!(merged_config)

    merged_config
    |> Map.update!(:quantiles, &Quantiles.normalize_config!/1)
    |> Model.new()
  end

  @seasonality_modes [:additive, :multiplicative]

  defp validate_config!(config) do
    validate_seasonality_mode!(config)
    validate_regressors!(config)
    validate_lagged_regressors!(config)
    validate_forecast_steps!(config)
  end

  defp validate_lagged_regressors!(%{lagged_regressors: lagged}) when lagged == %{}, do: :ok

  defp validate_lagged_regressors!(%{lagged_regressors: lagged, ar: ar}) when is_map(lagged) do
    for {name, spec} <- lagged do
      unless is_binary(name) and match?(%{lags: lags} when is_integer(lags) and lags > 0, spec) do
        raise ArgumentError,
              "lagged_regressors must map column names to %{lags: positive_integer}, " <>
                "got #{inspect(name)} => #{inspect(spec)}"
      end
    end

    unless ar.enabled and ar.lags > 0 do
      raise ArgumentError,
            "lagged_regressors need auto-regression enabled with lags > 0, " <>
              "since their lag windows are built from the same forecast origins."
    end

    :ok
  end

  defp validate_lagged_regressors!(%{lagged_regressors: lagged}) do
    raise ArgumentError, "lagged_regressors must be a map, got #{inspect(lagged)}"
  end

  defp validate_seasonality_mode!(%{seasonality: %{mode: mode}})
       when mode in @seasonality_modes do
    :ok
  end

  defp validate_seasonality_mode!(%{seasonality: %{mode: mode}}) do
    raise ArgumentError,
          "seasonality.mode must be one of #{inspect(@seasonality_modes)}, got #{inspect(mode)}"
  end

  defp validate_regressors!(%{regressors: regressors}) when is_list(regressors) do
    for name <- regressors, not is_binary(name) do
      raise ArgumentError, "regressors must be column name strings, got #{inspect(name)}"
    end

    :ok
  end

  defp validate_regressors!(%{regressors: regressors}) do
    raise ArgumentError,
          "regressors must be a list of column names, got #{inspect(regressors)}"
  end

  defp validate_forecast_steps!(%{ar: %{forecast_steps: steps}})
       when not (is_integer(steps) and steps > 0) do
    raise ArgumentError, "ar.forecast_steps must be a positive integer, got #{inspect(steps)}"
  end

  defp validate_forecast_steps!(%{ar: %{forecast_steps: steps} = ar}) when steps > 1 do
    unless ar.enabled and ar.lags > 0 do
      raise ArgumentError,
            "ar.forecast_steps > 1 needs auto-regression enabled with lags > 0. " <>
              "Without lags every date is forecast directly, so there are no steps to pick from."
    end

    :ok
  end

  defp validate_forecast_steps!(_config), do: :ok

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
    Regressors.validate_columns!(data, LaggedRegressors.names(model.config))
    processed_data = Seasonality.add_fourier_features(data, "ds", model.config.seasonality)
    # Reorder columns to put y first
    processed_data = DataFrame.select(processed_data, ["y" | processed_data.names -- ["y"]])

    y_full = processed_data["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32})
    {y_full_normalized, y_mean, y_std} = normalize(Nx.new_axis(y_full, -1))
    y_full_normalized = Nx.flatten(y_full_normalized)

    # Every training row targets one date. Without AR that is every date in
    # order. With AR, rows come from (origin, step) pairs, see
    # AR.training_rows/3, and each date-based feature is gathered at the
    # row's target position so all inputs line up with the targets.
    dates = Series.to_list(processed_data["ds"])
    {trend_full, trend_metadata} = Trend.build_features(dates, model.config)
    seasonality = Seasonality.build_features(dates, model.config)

    {y_normalized, ar_inputs, target_indices} =
      if ar_enabled?(model) do
        forecast_steps = AR.forecast_steps(model.config)
        max_lags = max(model.config.ar.lags, LaggedRegressors.max_lags(model.config))

        rows =
          AR.training_rows(y_full_normalized, model.config.ar.lags, forecast_steps,
            max_lags: max_lags
          )

        ar_inputs =
          %{"ar" => rows.lagged}
          |> put_step_mask(rows.step_mask)
          |> put_lagged_regressors_training_input(
            model,
            data,
            rows.origin_indices,
            forecast_steps
          )

        {rows.targets, ar_inputs, rows.target_indices}
      else
        {Nx.new_axis(y_full_normalized, -1), %{}, Enum.to_list(0..(length(dates) - 1))}
      end

    index_tensor = Nx.tensor(target_indices)
    dates_by_index = List.to_tuple(dates)
    feature_dates = Enum.map(target_indices, &elem(dates_by_index, &1))

    x =
      %{
        "trend" => Nx.take(trend_full, index_tensor, axis: 0),
        "yearly" => Nx.take(seasonality.yearly, index_tensor, axis: 0),
        "weekly" => Nx.take(seasonality.weekly, index_tensor, axis: 0)
      }
      |> Map.merge(ar_inputs)
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
      y_normalized: Nx.to_flat_list(y_full_normalized),
      lagged_regressors:
        Map.new(LaggedRegressors.names(model.config), fn name ->
          {name, LaggedRegressors.values_by_date(data, name)}
        end)
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
        observation and the forecast, since those get predicted too. Lagged
        regressor columns go in the same dataframe; they are only read up to
        each forecast origin, and training values are remembered, so the
        dataframe only needs to add what happened after training.

  ## Auto-regression and future dates

    When AR is enabled, each prediction needs the `lags` values ending at
    its origin. Dates inside the training data (or `:history`) are forecast
    one step ahead of the day before them, from observed values. Dates past
    the last observation are forecast in blocks of `ar.forecast_steps`: the
    first block directly from the last observation (step 1, 2, ... ahead),
    the next block from the end of the first, using its predictions as lags,
    and so on up to the latest requested date. Within a block there is no
    error compounding; across blocks there is, so far-out AR forecasts
    revert toward the level the model learned. This assumes daily, gap-free
    data.

  ## Returns

    A map with the combined prediction and each component: `:trend`,
    `:yearly_seasonality`, `:weekly_seasonality`, `:ar`, `:events`,
    `:regressors` and `:lagged_regressors`, plus `:quantiles`, a map from each configured quantile
    to its forecast (empty when none are configured). Quantile forecasts
    are clipped so an upper quantile is never below `:combined` and a lower
    one never above it.

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
        regressors: #Nx.Tensor<...>,
        quantiles: %{0.1 => #Nx.Tensor<...>, 0.9 => #Nx.Tensor<...>}
      }

  """
  @spec predict_components(Soothsayer.Model.t(), Explorer.Series.t(), keyword()) :: %{
          combined: Nx.Tensor.t(),
          trend: Nx.Tensor.t(),
          yearly_seasonality: Nx.Tensor.t(),
          weekly_seasonality: Nx.Tensor.t(),
          ar: Nx.Tensor.t(),
          events: Nx.Tensor.t(),
          regressors: Nx.Tensor.t(),
          lagged_regressors: Nx.Tensor.t(),
          quantiles: %{float() => Nx.Tensor.t()}
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
        observed_values = known_values(model, history)
        last_observed_date = observed_values |> Map.keys() |> Enum.max(Date)

        lagged_regressor_values =
          LaggedRegressors.known_values(model.config.training_data, regressors_df, model.config)

        known_values =
          forecast_missing_values(
            observed_values,
            lagged_regressor_values,
            model,
            prediction_dates,
            events_df,
            regressors_df
          )

        forecast_steps = AR.forecast_steps(model.config)

        {origin_dates, step_numbers} =
          prediction_dates
          |> Enum.map(&AR.origin_and_step(&1, last_observed_date, forecast_steps))
          |> Enum.unzip()

        Map.merge(
          x_input,
          lag_inputs(model, known_values, lagged_regressor_values, origin_dates, step_numbers)
        )
      else
        x_input
      end

    x_normalized = normalize_with_params(x_input, model.config.normalization.x)

    {quantile_outputs, predictions} = Map.pop(Model.predict(model, x_normalized), :quantiles)
    components = denormalize_components(predictions, model.config.normalization.y)

    Map.put(
      components,
      :quantiles,
      denormalize_quantiles(quantile_outputs, model.config.quantiles, components.combined, model)
    )
  end

  defp denormalize_quantiles(nil, _quantiles, _combined, _model), do: %{}

  defp denormalize_quantiles(outputs, quantiles, combined, model) do
    %{mean: mean, std: std} = model.config.normalization.y

    quantiles
    |> Enum.zip(Tuple.to_list(outputs))
    |> Map.new(fn {quantile, tensor} ->
      forecast = Nx.add(Nx.multiply(tensor, std), mean)

      clipped =
        if quantile > 0.5, do: Nx.max(forecast, combined), else: Nx.min(forecast, combined)

      {quantile, clipped}
    end)
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

  defp put_step_mask(inputs, nil), do: inputs
  defp put_step_mask(inputs, step_mask), do: Map.put(inputs, "forecast_step", step_mask)

  defp put_lagged_regressors_training_input(inputs, model, data, origin_indices, forecast_steps) do
    case LaggedRegressors.names(model.config) do
      [] ->
        inputs

      _names ->
        rows =
          LaggedRegressors.build_training_rows(data, model.config, origin_indices, forecast_steps)

        Map.put(inputs, "lagged_regressors", rows)
    end
  end

  # Every input that depends on a forecast origin: the target's own lags,
  # the lagged regressors' windows and, with more than one forecast step,
  # the one-hot step mask. Used for both the block rollout and the final
  # component pass, so a date gets the same value however it is requested.
  defp lag_inputs(model, known_values, lagged_regressor_values, origin_dates, step_numbers) do
    inputs =
      %{"ar" => AR.build_input(known_values, origin_dates, model.config.ar.lags)}
      |> put_step_mask(AR.step_mask(step_numbers, AR.forecast_steps(model.config)))

    case LaggedRegressors.names(model.config) do
      [] ->
        inputs

      _names ->
        rows = LaggedRegressors.build_input(lagged_regressor_values, origin_dates, model.config)
        Map.put(inputs, "lagged_regressors", rows)
    end
  end

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

  # Forecasts the days between the last observed date and the latest
  # prediction date in blocks of forecast_steps, each block predicted
  # directly from the block's origin, and records the predictions as known
  # values so later blocks can use them as lags. Returns the extended map.
  # Nothing is stored on the model.
  defp forecast_missing_values(
         known_values,
         lagged_regressor_values,
         model,
         prediction_dates,
         events_df,
         regressors_df
       ) do
    last_observed_date = known_values |> Map.keys() |> Enum.max(Date)
    last_prediction_date = Enum.max(prediction_dates, Date)

    if Date.compare(last_prediction_date, last_observed_date) == :gt do
      Date.range(Date.add(last_observed_date, 1), last_prediction_date)
      |> Enum.chunk_every(AR.forecast_steps(model.config))
      |> Enum.reduce(known_values, fn block_dates, known_values ->
        forecast_block(
          known_values,
          lagged_regressor_values,
          model,
          block_dates,
          events_df,
          regressors_df
        )
      end)
    else
      known_values
    end
  end

  # Predicts one block of consecutive dates from the day before the first
  # one. Returns known_values with the block's predictions added, in
  # normalized y space so they can feed later lags.
  defp forecast_block(
         known_values,
         lagged_regressor_values,
         model,
         block_dates,
         events_df,
         regressors_df
       ) do
    origin_dates = List.duplicate(Date.add(hd(block_dates), -1), length(block_dates))
    step_numbers = Enum.to_list(1..length(block_dates))

    inputs =
      model
      |> build_time_inputs(block_dates, events_df, regressors_df)
      |> Map.merge(
        lag_inputs(model, known_values, lagged_regressor_values, origin_dates, step_numbers)
      )
      |> normalize_with_params(model.config.normalization.x)

    %{combined: combined} = Model.predict(model, inputs)

    block_dates
    |> Enum.zip(Nx.to_flat_list(combined))
    |> Enum.reduce(known_values, fn {date, value}, acc -> Map.put(acc, date, value) end)
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

  # Inputs that are masks rather than measurements and must not be z-scored.
  @unnormalized_inputs ["forecast_step"]

  defp normalize_inputs(x) do
    Enum.reduce(x, {%{}, %{}}, fn {key, tensor}, acc ->
      normalize_single_input(key, tensor, acc)
    end)
  end

  defp normalize_single_input(key, tensor, {normalized, norm_params})
       when key in @unnormalized_inputs do
    {Map.put(normalized, key, tensor), norm_params}
  end

  defp normalize_single_input(key, tensor, {normalized, norm_params}) do
    {normalized_tensor, mean, std} = normalize(tensor)
    norm_param = %{mean: mean, std: std}

    {Map.put(normalized, key, normalized_tensor), Map.put(norm_params, key, norm_param)}
  end

  defp normalize_with_params(x, norm_params) do
    Map.new(x, fn
      {key, tensor} when key in @unnormalized_inputs ->
        {key, tensor}

      {key, tensor} ->
        %{mean: mean, std: std} = norm_params[key]
        {key, Nx.divide(Nx.subtract(tensor, mean), std)}
    end)
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
  Evaluates a model configuration with a rolling-origin backtest.

  Holds out the last `validation_fraction` of `data`, fits on the rest, and
  forecasts `horizon` steps ahead from every validation origin using only
  what was observed up to it. Returns the fitted model, overall and per-step
  MAE and RMSE, and a dataframe of every forecast. See `Soothsayer.Backtest`.

  ## Examples

      iex> result = Soothsayer.backtest(Soothsayer.new(%{ar: %{enabled: true, lags: 14, forecast_steps: 7}}), df)
      iex> result.metrics
      %{mean_absolute_error: 5.65, root_mean_squared_error: 7.04}
      iex> result.by_step[7].mean_absolute_error
      6.9

  """
  @spec backtest(Soothsayer.Model.t(), Explorer.DataFrame.t(), keyword()) ::
          Soothsayer.Backtest.result()
  def backtest(%Model{} = model, %DataFrame{} = data, opts \\ []) do
    Soothsayer.Backtest.run(model, data, opts)
  end

  @doc """
  Extracts the raw AR layer weights from a fitted model.

  For linear AR models, returns the output layer weights.
  For deep AR-Net models, returns all layer weights including hidden layers.

  The output kernel has shape `{inputs, forecast_steps}`: row `i` is the
  i-th oldest lag and column `s` holds the weights for step `s + 1` ahead.

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
