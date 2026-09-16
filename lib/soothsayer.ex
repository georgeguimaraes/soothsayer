defmodule Soothsayer do
  @moduledoc """
  The main module for the Soothsayer library, providing functions for creating, fitting, and using time series forecasting models.
  """

  require Logger

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Events
  alias Soothsayer.Frequency
  alias Soothsayer.Holidays
  alias Soothsayer.LaggedRegressors
  alias Soothsayer.MissingData
  alias Soothsayer.Model
  alias Soothsayer.Quantiles
  alias Soothsayer.Regressors
  alias Soothsayer.Seasonality
  alias Soothsayer.Timestamp
  alias Soothsayer.Trainer
  alias Soothsayer.Trend

  @doc """
  Creates a new Soothsayer model with the given configuration.

  ## Parameters

    * `config` - A map containing the model configuration. Defaults to an empty map.

  ## Returns

    A new `Soothsayer.Model` struct.

  ## Examples

      iex> Soothsayer.new()
      %Soothsayer.Model{config: %{trend: %{enabled: true}, seasonality: %{yearly: %{enabled: true, fourier_terms: 6}, weekly: %{enabled: true, fourier_terms: 3}}, epochs: :auto, learning_rate: :auto, schedule: :one_cycle, ...}, network: %Axon.Node{}, params: nil, predict_fn: nil}

      iex> Soothsayer.new(%{epochs: 200, learning_rate: 0.005})
      %Soothsayer.Model{config: %{trend: %{enabled: true}, seasonality: %{yearly: %{enabled: true, fourier_terms: 6}, weekly: %{enabled: true, fourier_terms: 3}}, epochs: 200, learning_rate: 0.005}, network: %Axon.Node{}, params: nil, predict_fn: nil}

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
        weekly: %{enabled: true, fourier_terms: 3},
        daily: %{enabled: :auto, fourier_terms: 6}
      },
      frequency: :auto,
      ar: %{enabled: false, lags: 0, layers: [], regularization: nil, forecast_steps: 1},
      events: %{},
      holidays: %{
        countries: [],
        steps_before: 0,
        steps_after: 0,
        regions: [],
        include_informal: false
      },
      regressors: [],
      lagged_regressors: %{},
      quantiles: [],
      missing: %{impute: true, impute_linear: 10, impute_rolling: 10, drop_samples: false},
      epochs: :auto,
      learning_rate: :auto,
      schedule: :one_cycle,
      optimizer: :adam,
      batch_size: nil,
      seed: nil
    }

    merged_config = deep_merge(default_config, config)
    validate_config!(merged_config)

    merged_config
    |> Map.update!(:quantiles, &Quantiles.normalize_config!/1)
    |> Map.update!(:holidays, &Holidays.normalize_config!/1)
    |> Map.update!(:events, &fill_event_defaults/1)
    |> Model.new()
  end

  @seasonality_modes [:additive, :multiplicative]

  defp validate_config!(config) do
    validate_seasonality_mode!(config)
    validate_seasonality_enabled!(config)
    Frequency.validate!(config.frequency)
    validate_regressors!(config)
    validate_lagged_regressors!(config)
    validate_forecast_steps!(config)
    validate_missing!(config)
    validate_events!(config)
    validate_training_options!(config)
  end

  @event_defaults %{steps_before: 0, steps_after: 0}

  defp fill_event_defaults(events) do
    Map.new(events, fn {name, spec} -> {name, Map.merge(@event_defaults, spec)} end)
  end

  defp validate_events!(%{events: events}) when is_map(events) do
    for {name, spec} <- events do
      unless is_binary(name) and is_map(spec) do
        raise ArgumentError,
              "events must map names to %{steps_before: count, steps_after: count}, " <>
                "got #{inspect(name)} => #{inspect(spec)}"
      end

      if Map.has_key?(spec, :lower_window) or Map.has_key?(spec, :upper_window) do
        raise ArgumentError,
              "events.#{name} uses steps_before and steps_after now, both counts >= 0 " <>
                "(lower_window: -2, upper_window: 1 becomes steps_before: 2, steps_after: 1)"
      end

      for key <- [:steps_before, :steps_after],
          value = Map.get(spec, key, 0),
          not (is_integer(value) and value >= 0) do
        raise ArgumentError,
              "events.#{name}.#{key} must be an integer >= 0, got #{inspect(value)}"
      end

      unless spec[:recurring] in [nil, :yearly] do
        raise ArgumentError,
              "events.#{name}.recurring must be :yearly or left out, got #{inspect(spec[:recurring])}"
      end
    end

    :ok
  end

  defp validate_events!(%{events: events}) do
    raise ArgumentError, "events must be a map of event names to windows, got #{inspect(events)}"
  end

  defp validate_missing!(%{missing: missing}) do
    for key <- [:impute, :drop_samples], value = Map.get(missing, key), not is_boolean(value) do
      raise ArgumentError, "missing.#{key} must be true or false, got #{inspect(value)}"
    end

    for key <- [:impute_linear, :impute_rolling],
        value = Map.get(missing, key),
        not (is_integer(value) and value >= 0) do
      raise ArgumentError,
            "missing.#{key} must be a non-negative integer, got #{inspect(value)}"
    end

    :ok
  end

  defp validate_training_options!(config) do
    unless config.learning_rate == :auto or
             (is_number(config.learning_rate) and config.learning_rate > 0) do
      raise ArgumentError,
            "learning_rate must be a positive number or :auto, got #{inspect(config.learning_rate)}"
    end

    unless config.epochs == :auto or (is_integer(config.epochs) and config.epochs > 0) do
      raise ArgumentError,
            "epochs must be a positive integer or :auto, got #{inspect(config.epochs)}"
    end

    unless config.schedule in [:constant, :one_cycle] do
      raise ArgumentError,
            "schedule must be :constant or :one_cycle, got #{inspect(config.schedule)}"
    end

    unless config.optimizer in [:adam, :adamw] do
      raise ArgumentError, "optimizer must be :adam or :adamw, got #{inspect(config.optimizer)}"
    end

    :ok
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

  defp validate_seasonality_enabled!(%{seasonality: seasonality}) do
    for period <- Seasonality.periods(),
        enabled = get_in(seasonality, [period, :enabled]),
        enabled not in [true, false, :auto] do
      raise ArgumentError,
            "seasonality.#{period}.enabled must be true, false or :auto, got #{inspect(enabled)}"
    end

    :ok
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
    * `data` - An `Explorer.DataFrame` with a "ds" column of dates or naive
      datetimes, strictly increasing, and a numeric "y" column.
    * `opts` - Optional keyword list:
      - `:events` - An `Explorer.DataFrame` with "event" and "ds" columns,
        one row per occurrence of each configured event. The dates are
        remembered by the model, so predicting inside the training period
        doesn't need them again. Events with `recurring: :yearly` repeat
        every year on the same month and day.

    When the model config lists `regressors`, `data` must contain a column
    for each of them. With `holidays: %{countries: [...]}` every holiday of
    those countries becomes an event of its own, named as holidefs names it,
    see `Soothsayer.Holidays`; the fitted `config.events` lists them.

  ## Missing data

    Missing values (`nil` or NaN) and missing rows are handled the way
    NeuralProphet does, see `Soothsayer.MissingData` and the missing data
    guide. Without auto-regression the rows with a missing `y` are dropped.
    With auto-regression the data is put on the frequency grid, trailing
    gaps are dropped and the rest are imputed, linearly up to
    `missing.impute_linear` values from each side of a gap and then with a
    rolling mean over `missing.impute_rolling` more. Gaps left open after
    that raise, unless `missing: %{drop_samples: true}` skips the training
    samples that touch them.

  ## Frequency and seasonality

    With `frequency: :auto` the step between rows is inferred from the most
    common gap in "ds" (daily, hourly, every 5 minutes, monthly, ...) and
    stored on the fitted model as `config.frequency`. Auto-regression lags,
    forecast blocks and event windows all move by that step. Every
    seasonality with `enabled: :auto` is then switched on or off from the
    data span and step, see `Soothsayer.Seasonality.resolve_auto/3`.

  ## Returns

    An updated `Soothsayer.Model` struct with fitted parameters.

  ## Examples

      iex> model = Soothsayer.new()
      iex> data = Explorer.DataFrame.new(%{"ds" => [...], "y" => [...]})
      iex> fitted_model = Soothsayer.fit(model, data)
      %Soothsayer.Model{config: %{}, network: %Axon.Node{}, params: %{}, predict_fn: #Function<...>}

      iex> events_df = Explorer.DataFrame.new(%{"event" => ["sale"], "ds" => [~D[2023-01-01]]})
      iex> fitted_model = Soothsayer.fit(model, data, events: events_df)
      %Soothsayer.Model{config: %{}, network: %Axon.Node{}, params: %{}, predict_fn: #Function<...>}

  """
  @spec fit(Soothsayer.Model.t(), Explorer.DataFrame.t(), keyword()) :: Soothsayer.Model.t()
  def fit(%Model{} = model, %DataFrame{} = data, opts \\ []) do
    events_df = Keyword.get(opts, :events)
    validate_training_data!(data)
    Regressors.validate_columns!(data, model.config.regressors)
    Regressors.validate_columns!(data, LaggedRegressors.names(model.config))

    timestamps = Timestamp.from_series(data["ds"])
    Timestamp.validate_sorted!(timestamps)
    frequency = resolve_frequency(model.config.frequency, timestamps)

    # Missing rows and values are dropped, regridded or imputed before
    # anything is built, so every component sees the same rows. Whatever is
    # still missing is NaN in `data` and listed in `unfilled` by row.
    {data, unfilled} = MissingData.prepare(data, model.config, frequency)
    timestamps = Timestamp.from_series(data["ds"])

    # The frequency and the :auto seasonalities are settled before anything
    # is built from the config, so every component sees the same answer.
    config =
      model.config
      |> Map.put(:frequency, frequency)
      |> Map.update!(:seasonality, &Seasonality.resolve_auto(&1, timestamps, frequency))
      |> put_holiday_events(timestamps)

    model = %{model | config: config}

    # The mean and std come from the known values only, so the NaNs left at
    # unfilled positions stay NaN and never reach a training sample.
    y_values = data["y"] |> Series.cast({:f, 64}) |> Series.to_list()
    y_full = Nx.tensor(y_values, type: {:f, 32})
    known_y = y_values |> Enum.reject(&(&1 == :nan)) |> Nx.tensor(type: {:f, 32})
    {_known_normalized, y_mean, y_std} = normalize(Nx.new_axis(known_y, -1))
    y_full_normalized = Nx.divide(Nx.subtract(y_full, y_mean), y_std)

    # One training sample per forecast origin. Without AR that is every
    # timestamp on its own. With AR a sample is the origin's lags followed
    # by its forecast_steps targets, see AR.training_samples/4, and every
    # time-based feature is gathered at all of those positions so the
    # network can evaluate the components at the lag timestamps too.
    {y_normalized, ar_inputs, position_indices} =
      if ar_enabled?(model) do
        max_lags = max(config.ar.lags, LaggedRegressors.max_lags(config))
        skip_positions = unfilled |> Map.values() |> Enum.reduce(MapSet.new(), &MapSet.union/2)

        samples =
          AR.training_samples(y_full_normalized, config.ar.lags, AR.forecast_steps(config),
            max_lags: max_lags,
            skip_positions: skip_positions
          )

        if samples.skipped_origins > 0 and not config.missing.drop_samples do
          MissingData.raise_unfilled!(samples.skipped_origins, config.missing)
        end

        if samples.skipped_origins > 0 do
          Logger.info(
            "Skipped #{samples.skipped_origins} training samples touching missing values"
          )
        end

        ar_inputs =
          %{"ar" => samples.lagged}
          |> put_lagged_regressors_training_input(model, data, samples.origin_indices)

        {samples.targets, ar_inputs, samples.position_indices}
      else
        {Nx.new_axis(y_full_normalized, -1), %{}, Nx.iota({length(timestamps), 1})}
      end

    {trend_full, trend_metadata} = Trend.build_features(timestamps, config)

    x =
      %{"trend" => trend_full}
      |> Map.merge(seasonality_inputs(timestamps, config))
      |> put_events_input(model, timestamps, events_df)
      |> put_training_regressors_input(model, timestamps, data)
      |> Map.new(fn {key, features} -> {key, Nx.take(features, position_indices, axis: 0)} end)
      |> Map.merge(ar_inputs)

    {x_normalized, x_norm} = normalize_inputs(x)

    # The network is rebuilt now that the y normalization is known, since
    # multiplicative seasonality needs the series level as a constant.
    config = Map.put(config, :normalization, %{x: x_norm, y: %{mean: y_mean, std: y_std}})

    network = Model.build_network(config)

    # :auto epochs and learning rate are resolved here so the fitted model
    # records the values that were actually used.
    config =
      config
      |> Map.update!(:epochs, fn
        :auto -> Trainer.auto_epochs(Nx.axis_size(y_normalized, 0))
        epochs -> epochs
      end)
      |> then(fn config ->
        Map.put(
          config,
          :learning_rate,
          Trainer.resolve_learning_rate(network, x_normalized, y_normalized, config)
        )
      end)

    model = %{model | config: config, network: network}

    fitted_model = Model.fit(model, x_normalized, y_normalized, config.epochs)

    # Store training data for prediction lookups. Regressor values are kept
    # so the lag positions of a forecast can reach into the training period.
    y_normalized_values = Nx.to_flat_list(y_full_normalized)

    known_values =
      timestamps
      |> Enum.zip(y_normalized_values)
      |> Enum.reject(fn {_timestamp, value} -> value == :nan end)
      |> Map.new()

    training_data = %{
      timestamps: timestamps,
      y_normalized: y_normalized_values,
      known_values: known_values,
      last_timestamp: Enum.max(timestamps, NaiveDateTime),
      regressors: Map.new(config.regressors, &{&1, Regressors.values_by_timestamp(data, &1)}),
      lagged_regressors:
        Map.new(LaggedRegressors.names(config), &{&1, Regressors.values_by_timestamp(data, &1)}),
      event_dates: Events.frame_dates(events_df)
    }

    %{
      fitted_model
      | config:
          config
          |> Map.put(:training_data, training_data)
          |> Map.put(:first_timestamp, trend_metadata.first_timestamp)
          |> Map.put(:changepoint_positions, trend_metadata.changepoint_positions)
    }
  end

  defp resolve_frequency(:auto, timestamps), do: Frequency.infer(timestamps)
  defp resolve_frequency(frequency, _timestamps), do: frequency

  @component_columns [
    :trend,
    :yearly_seasonality,
    :weekly_seasonality,
    :daily_seasonality,
    :ar,
    :events,
    :regressors,
    :lagged_regressors
  ]

  @doc """
  Makes predictions using a fitted Soothsayer model.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct.
    * `x` - An `Explorer.Series` of dates or naive datetimes to predict for.
    * `opts` - Optional keyword list:
      - `:events` - An `Explorer.DataFrame` with "event" and "ds" columns
        for occurrences the model doesn't know yet. Occurrences given at
        fit, yearly recurring events and country holidays apply without it.
      - `:history` - An `Explorer.DataFrame` with "ds" and "y" columns holding
        observations newer than the training data, sorted by "ds". Missing
        values in it are imputed like training data. Only used when
        auto-regression is enabled, see `predict_components/3`.
      - `:regressors` - An `Explorer.DataFrame` with "ds" plus one column per
        configured regressor, covering every predicted date. Required when
        the model was fitted with regressors.

  ## Returns

    An `Explorer.DataFrame` with one row per requested timestamp and the
    columns, in this order:

    * `"ds"` - the timestamps, the series `x` itself
    * `"yhat"` - the forecast (the median when quantiles are configured)
    * one column per configured quantile, `"yhat_10"` for `0.1`,
      `"yhat_97.5"` for `0.975`, ascending
    * `"trend"` - always present; a flat line at the training mean when the
      trend is disabled
    * one column per enabled component among `"yearly_seasonality"`,
      `"weekly_seasonality"`, `"daily_seasonality"`, `"ar"`, `"events"`,
      `"regressors"` and `"lagged_regressors"`

    The component columns add up to `"yhat"`. Use `predict_components/3`
    for the same values as tensors.

    With auto-regression, a timestamp whose lags are unknown (the first
    `ar.lags` timestamps of the training data, or the steps after a gap
    that couldn't be imputed) gets NaN for `"yhat"`, `"ar"` and the
    quantiles rather than a forecast built on made-up lags.

  ## Examples

      iex> fitted_model = Soothsayer.fit(model, training_data)
      iex> future_dates = Explorer.Series.from_list([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])
      iex> Soothsayer.predict(fitted_model, future_dates)
      #Explorer.DataFrame<
        Polars[3 x 5]
        ds date [2023-01-01, 2023-01-02, 2023-01-03]
        yhat f64 [1.5, 2.3, 3.1]
        trend f64 [1.4, 2.1, 2.8]
        yearly_seasonality f64 [0.2, 0.3, 0.4]
        weekly_seasonality f64 [-0.1, -0.1, -0.1]
      >

  """
  @spec predict(Soothsayer.Model.t(), Explorer.Series.t(), keyword()) :: Explorer.DataFrame.t()
  def predict(%Model{} = model, %Series{} = x, opts \\ []) do
    components = predict_components(model, x, opts)
    rows = Series.size(x)

    quantile_columns =
      components.quantiles
      |> Enum.sort()
      |> Enum.map(fn {quantile, tensor} -> {quantile_column(quantile), tensor} end)

    # Disabled components are scalar zeros and stay out of the frame, except
    # the trend, which carries the level of the series even when disabled
    # and is needed for the columns to add up to yhat.
    component_columns =
      for key <- @component_columns,
          tensor = components[key],
          key == :trend or Nx.rank(tensor) == 2,
          do: {Atom.to_string(key), tensor}

    columns =
      [{"yhat", components.combined}] ++ quantile_columns ++ component_columns

    DataFrame.new([
      {"ds", x} | Enum.map(columns, fn {name, tensor} -> {name, column(tensor, rows)} end)
    ])
  end

  # A disabled trend is a single value (the training mean), broadcast to
  # every row.
  defp column(tensor, rows) do
    values =
      case Nx.size(tensor) do
        1 -> tensor |> Nx.reshape({}) |> Nx.to_number() |> List.duplicate(rows)
        _size -> Nx.to_flat_list(tensor)
      end

    Series.from_list(values, dtype: {:f, 64})
  end

  # 0.1 -> "yhat_10", 0.975 -> "yhat_97.5"
  defp quantile_column(quantile) do
    percent = Float.round(quantile * 100, 1)

    label =
      if percent == Float.floor(percent),
        do: Integer.to_string(trunc(percent)),
        else: :erlang.float_to_binary(percent, decimals: 1)

    "yhat_" <> label
  end

  @doc """
  Makes predictions and returns the individual components (trend, seasonality) using a fitted Soothsayer model.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct.
    * `x` - An `Explorer.Series` of dates or naive datetimes to predict for.
      Plain dates mean midnight, which matters for sub-daily models.
    * `opts` - Optional keyword list:
      - `:events` - An `Explorer.DataFrame` with "event" and "ds" columns
        for occurrences the model doesn't know yet. Occurrences given at
        fit, yearly recurring events and country holidays apply without it.
      - `:history` - An `Explorer.DataFrame` with "ds" and "y" columns holding
        observations newer than the training data, sorted by "ds". Missing
        values in it are imputed like training data, see
        `Soothsayer.MissingData`. Only used when auto-regression is enabled.
      - `:regressors` - An `Explorer.DataFrame` with "ds" plus one column per
        configured regressor. Required when the model was fitted with
        regressors, and it must cover every predicted date. With
        auto-regression it must also cover the steps between the last
        observation and the forecast, since those get predicted too, but
        not the rest of the last forecast block past the latest requested
        date. Training values are remembered, so the dataframe only needs
        to add what happened after training. Lagged regressor columns go in
        the same dataframe and are only read up to each forecast origin.

  ## Auto-regression and future timestamps

    When AR is enabled, each prediction needs the `lags` values ending at
    its origin. Timestamps inside the training data (or `:history`) are
    forecast one step ahead of the step before them, from observed values.
    Timestamps past the last observation are forecast in blocks of
    `ar.forecast_steps`: the first block directly from the last observation
    (step 1, 2, ... ahead), the next block from the end of the first, using
    its predictions as lags, and so on up to the latest requested
    timestamp. Within a block there is no error compounding; across blocks
    there is, so far-out AR forecasts revert toward the level the model
    learned. Steps are steps of the model's frequency. Training data and
    history are put on that grid at fit (see `Soothsayer.MissingData`), and
    every requested timestamp must sit on it too (an hourly model can't
    forecast half past the hour).

  ## Returns

    A map with the combined prediction and each component: `:trend`,
    `:yearly_seasonality`, `:weekly_seasonality`, `:daily_seasonality`,
    `:ar`, `:events`, `:regressors` and `:lagged_regressors`, plus
    `:quantiles`, a map from each configured quantile
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
        daily_seasonality: #Nx.Tensor<...>,
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
          daily_seasonality: Nx.Tensor.t(),
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

    prediction_timestamps = Timestamp.from_series(x)
    training_data = model.config.training_data

    regressor_values = %{
      regressors: Regressors.known_values(training_data, regressors_df, model.config.regressors),
      lagged_regressors: LaggedRegressors.known_values(training_data, regressors_df, model.config)
    }

    {x_input, step_numbers} =
      if ar_enabled?(model) do
        {observed_values, last_observed} = known_values(model, history)
        forecast_steps = AR.forecast_steps(model.config)

        known_values =
          forecast_missing_values(
            observed_values,
            last_observed,
            regressor_values,
            model,
            prediction_timestamps,
            events_df
          )

        {origins, step_numbers} =
          prediction_timestamps
          |> Enum.map(
            &AR.origin_and_step(&1, last_observed, forecast_steps, model.config.frequency)
          )
          |> Enum.unzip()

        inputs =
          sample_inputs(
            model,
            origins,
            prediction_timestamps,
            known_values,
            regressor_values,
            events_df
          )

        {inputs, step_numbers}
      else
        samples = Enum.map(prediction_timestamps, &[&1])
        required = MapSet.new(prediction_timestamps)

        inputs =
          build_time_inputs(model, samples, required, events_df, regressor_values.regressors)

        {inputs, List.duplicate(1, length(prediction_timestamps))}
      end

    x_normalized = normalize_with_params(x_input, model.config.normalization.x)

    # The network forecasts every step from each sample's origin. Each
    # requested timestamp keeps the step it was asked for, so every
    # component comes out as {n, 1} whatever forecast_steps is.
    step_index = step_numbers |> Enum.map(&(&1 - 1)) |> Nx.tensor() |> Nx.reshape({:auto, 1})

    {quantile_outputs, predictions} =
      Model.predict(model, x_normalized)
      |> Map.new(fn {key, output} -> {key, select_step(output, step_index)} end)
      |> Map.pop(:quantiles)

    components = denormalize_components(predictions, model.config.normalization.y)

    Map.put(
      components,
      :quantiles,
      denormalize_quantiles(quantile_outputs, model.config.quantiles, components.combined, model)
    )
  end

  defp select_step(outputs, step_index) when is_tuple(outputs) do
    outputs |> Tuple.to_list() |> Enum.map(&select_step(&1, step_index))
  end

  # Disabled components are scalar zeros and stay that way
  defp select_step(output, step_index) do
    if Nx.rank(output) == 2, do: Nx.take_along_axis(output, step_index, axis: 1), else: output
  end

  defp denormalize_quantiles(nil, _quantiles, _combined, _model), do: %{}

  defp denormalize_quantiles(outputs, quantiles, combined, model) do
    %{mean: mean, std: std} = model.config.normalization.y

    quantiles
    |> Enum.zip(outputs)
    |> Map.new(fn {quantile, tensor} ->
      forecast = Nx.add(Nx.multiply(tensor, std), mean)

      clipped =
        if quantile > 0.5, do: Nx.max(forecast, combined), else: Nx.min(forecast, combined)

      {quantile, clipped}
    end)
  end

  # Built whenever events are configured, with or without a dataframe: the
  # dates come from the frame, from what the model remembers, from yearly
  # recurrence and from the country holidays.
  defp put_events_input(x, model, timestamps, events_df) do
    events_config = model.config[:events] || %{}

    if map_size(events_config) > 0 do
      event_dates = Events.event_dates(events_df, model.config, timestamps)

      events_input =
        Events.build_features(
          Series.from_list(timestamps),
          event_dates,
          events_config,
          model.config.frequency
        )

      Map.put(x, "events", events_input)
    else
      x
    end
  end

  # Every holiday of the configured countries becomes an event with the
  # shared holiday window. The names found in the training years are what
  # the network gets columns for.
  defp put_holiday_events(%{holidays: %{countries: []}} = config, _timestamps), do: config

  defp put_holiday_events(config, timestamps) do
    names = Holidays.names(config.holidays, Events.years(timestamps))

    case Enum.filter(names, &Map.has_key?(config.events, &1)) do
      [] ->
        :ok

      taken ->
        raise ArgumentError,
              "#{inspect(taken)} are both configured events and country holidays. " <>
                "Rename the events or leave the holidays to the holidays config."
    end

    window = %{
      steps_before: config.holidays.steps_before,
      steps_after: config.holidays.steps_after
    }

    config
    |> Map.update!(:events, &Map.merge(&1, Map.new(names, fn name -> {name, window} end)))
    |> put_in([:holidays, :names], names)
  end

  defp seasonality_inputs(timestamps, config) do
    timestamps
    |> Seasonality.build_features(config)
    |> Map.new(fn {period, features} -> {Atom.to_string(period), features} end)
  end

  defp put_training_regressors_input(x, model, timestamps, data) do
    case model.config.regressors do
      [] -> x
      names -> Map.put(x, "regressors", Regressors.build_features(timestamps, data, names))
    end
  end

  # Regressor values must exist at the required timestamps (the lag
  # positions and whatever was asked for). Other target positions of a
  # sample get the training mean, which normalizes to zero, since a value
  # there only feeds that position's own output.
  defp put_regressors_input(x, model, timestamps, regressor_values, required_timestamps) do
    case model.config.regressors do
      [] ->
        x

      names ->
        fill = model.config.normalization.x["regressors"].mean |> Nx.to_flat_list()

        features =
          Regressors.build_features(timestamps, regressor_values, names,
            required: required_timestamps,
            fill: fill
          )

        Map.put(x, "regressors", features)
    end
  end

  defp ar_enabled?(model), do: AR.lags(model.config) > 0

  defp put_lagged_regressors_training_input(inputs, model, data, origin_indices) do
    case LaggedRegressors.names(model.config) do
      [] ->
        inputs

      _names ->
        rows = LaggedRegressors.build_training_rows(data, model.config, origin_indices)
        Map.put(inputs, "lagged_regressors", rows)
    end
  end

  # The inputs for one sample per origin: the time-based features at the
  # sample's lag and target positions, the lags themselves and the lagged
  # regressor windows. Used for both the block rollout and the final
  # component pass, so a date gets the same value however it is requested.
  defp sample_inputs(
         model,
         origins,
         requested_timestamps,
         known_values,
         regressor_values,
         events_df
       ) do
    config = model.config
    lags = AR.lags(config)

    samples =
      Enum.map(
        origins,
        &AR.sample_timestamps(&1, lags, AR.forecast_steps(config), config.frequency)
      )

    required =
      samples
      |> Enum.flat_map(&Enum.take(&1, lags))
      |> Enum.concat(requested_timestamps)
      |> MapSet.new(&Timestamp.to_naive_datetime/1)

    model
    |> build_time_inputs(samples, required, events_df, regressor_values.regressors)
    |> Map.merge(lag_inputs(model, known_values, regressor_values.lagged_regressors, origins))
  end

  # Every input that depends on a forecast origin: the target's own lags
  # and the lagged regressors' windows.
  defp lag_inputs(model, known_values, lagged_regressor_values, origins) do
    frequency = model.config.frequency
    inputs = %{"ar" => AR.build_input(known_values, origins, model.config.ar.lags, frequency)}

    case LaggedRegressors.names(model.config) do
      [] ->
        inputs

      _names ->
        rows =
          LaggedRegressors.build_input(lagged_regressor_values, origins, model.config, frequency)

        Map.put(inputs, "lagged_regressors", rows)
    end
  end

  defp validate_regressors_option!(%Model{config: %{regressors: []}}, _regressors_df), do: :ok

  defp validate_regressors_option!(%Model{config: %{regressors: names}}, nil) do
    raise ArgumentError,
          "This model was fitted with regressors #{inspect(names)}. " <>
            "Pass regressors: a dataframe with \"ds\" and those columns to predict."
  end

  defp validate_regressors_option!(%Model{config: %{regressors: names}}, %DataFrame{} = frame) do
    Regressors.validate_columns!(frame, names)
  end

  # Builds every input that depends only on the timestamp (trend,
  # seasonality, events and regressors) for a list of samples, each a list
  # of `positions` timestamps, as {samples, positions, features} tensors.
  # AR is added separately since it depends on previous values.
  defp build_time_inputs(model, samples, required_timestamps, events_df, regressor_values) do
    config = model.config
    timestamps = List.flatten(samples)
    positions = AR.positions(config)

    t = Trend.date_to_numeric(timestamps, config.first_timestamp) |> Nx.new_axis(-1)
    changepoint_features = Trend.build_changepoint_features(t, config.changepoint_positions)
    trend_input = Trend.build_trend_input(t, changepoint_features)

    %{"trend" => trend_input}
    |> Map.merge(seasonality_inputs(timestamps, config))
    |> put_events_input(model, timestamps, events_df)
    |> put_regressors_input(model, timestamps, regressor_values, required_timestamps)
    |> Map.new(fn {key, features} ->
      {key, Nx.reshape(features, {length(samples), positions, :auto})}
    end)
  end

  # Observed values in normalized y space, keyed by timestamp, and the last
  # observed timestamp. Training data comes first, then `history` overrides
  # or extends it.
  defp known_values(model, nil) do
    training_data = model.config.training_data
    {AR.known_values(training_data), training_data.last_timestamp}
  end

  # History gets the same treatment as training data with lags: put on the
  # frequency grid, trailing gap dropped, the rest imputed. Values still
  # missing stay unknown, and a forecast whose lags need them is NaN. The
  # last observed timestamp is the last one with a known value, so a
  # trailing gap is forecast like the future rather than looked up.
  defp known_values(model, %DataFrame{} = history) do
    validate_history!(history)
    %{mean: mean, std: std} = model.config.normalization.y
    mean = mean |> Nx.squeeze() |> Nx.to_number()
    std = std |> Nx.squeeze() |> Nx.to_number()

    history_timestamps = Timestamp.from_series(history["ds"])
    Timestamp.validate_sorted!(history_timestamps)

    {history_timestamps, history_values} =
      MissingData.fill_history(
        history_timestamps,
        Series.to_list(history["y"]),
        model.config.frequency,
        model.config.missing
      )

    history_known =
      for {timestamp, value} <- Enum.zip(history_timestamps, history_values),
          not is_nil(value),
          into: %{},
          do: {timestamp, (value - mean) / std}

    {training_values, last_training_timestamp} = known_values(model, nil)

    {Map.merge(training_values, history_known),
     Enum.max([last_training_timestamp | Map.keys(history_known)], NaiveDateTime)}
  end

  # Forecasts the steps between the last observed timestamp and the latest
  # prediction timestamp in blocks of forecast_steps, each block predicted
  # directly from the block's origin, and records the predictions as known
  # values so later blocks can use them as lags. Returns the extended map.
  # Nothing is stored on the model.
  defp forecast_missing_values(
         known_values,
         last_observed,
         regressor_values,
         model,
         prediction_timestamps,
         events_df
       ) do
    last_prediction = Enum.max(prediction_timestamps, NaiveDateTime)

    last_observed
    |> Frequency.range(last_prediction, model.config.frequency)
    |> Enum.chunk_every(AR.forecast_steps(model.config))
    |> Enum.reduce(known_values, fn block_timestamps, known_values ->
      forecast_block(known_values, regressor_values, model, block_timestamps, events_df)
    end)
  end

  # Predicts one block of consecutive timestamps from the step before the
  # first one, as a single sample whose targets are the whole block (the
  # last block may be shorter than forecast_steps and keeps only what it
  # needs). Returns known_values with the block's predictions added, in
  # normalized y space so they can feed later lags.
  defp forecast_block(known_values, regressor_values, model, block_timestamps, events_df) do
    origin = Frequency.shift(hd(block_timestamps), -1, model.config.frequency)

    inputs =
      model
      |> sample_inputs([origin], block_timestamps, known_values, regressor_values, events_df)
      |> normalize_with_params(model.config.normalization.x)

    %{combined: combined} = Model.predict(model, inputs)

    # A block whose lags reach into a gap that couldn't be imputed comes out
    # NaN. Those stay unknown, so everything forecast from them is NaN too.
    block_timestamps
    |> Enum.zip(Nx.to_flat_list(combined))
    |> Enum.reject(fn {_timestamp, value} -> value == :nan end)
    |> Enum.reduce(known_values, fn {timestamp, value}, acc -> Map.put(acc, timestamp, value) end)
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

  # {samples, positions, features} inputs are z-scored per feature over
  # samples and positions alike, so lag and target positions share the
  # statistics the shared dense layer expects.
  defp normalize(tensor) do
    axes = if Nx.rank(tensor) == 3, do: [0, 1], else: [0]
    mean = Nx.mean(tensor, axes: axes)
    std = Nx.standard_deviation(tensor, axes: axes)
    std = Nx.select(Nx.equal(std, 0), Nx.tensor(1), std)
    {Nx.divide(Nx.subtract(tensor, mean), std), mean, std}
  end

  # The lags are already in normalized y space, which is the space the
  # components subtracted from them predict in, so they must stay there.
  @unnormalized_inputs ["ar"]

  defp normalize_inputs(x) do
    Enum.reduce(x, {%{}, %{}}, fn {key, tensor}, acc ->
      normalize_single_input(key, tensor, acc)
    end)
  end

  defp normalize_single_input(key, tensor, {normalized, norm_params})
       when key in @unnormalized_inputs do
    {Map.put(normalized, key, tensor), norm_params}
  end

  # The trend features are scaled by the training span, not z-scored, so t
  # runs from 0 to 1 over the training data and each changepoint hinge
  # max(0, t - s) keeps those units, as in NeuralProphet. Z-scoring each
  # hinge on its own blows up the late ones (few nonzero values, tiny
  # standard deviation), which lets the slope of the last segment swing
  # with the last few days of data and then extrapolate that swing.
  defp normalize_single_input("trend" = key, tensor, {normalized, norm_params}) do
    features = Nx.axis_size(tensor, 2)
    span = tensor[[.., .., 0]] |> Nx.reduce_max()
    norm_param = %{mean: Nx.broadcast(0.0, {features}), std: Nx.broadcast(span, {features})}

    {Map.put(normalized, key, Nx.divide(tensor, span)), Map.put(norm_params, key, norm_param)}
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
              "Training data must contain a 'ds' (date or naive datetime) column. " <>
                "Available columns: #{inspect(columns)}"

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
  Hidden layers have a `:bias` as well; the output layer has none.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct with AR enabled.

  ## Returns

    A map of layer names to maps with a `:kernel` tensor and, for hidden
    layers, a `:bias` tensor.

  ## Examples

      iex> model = Soothsayer.new(%{ar: %{enabled: true, lags: 3}})
      iex> fitted_model = Soothsayer.fit(model, data)
      iex> weights = Soothsayer.get_ar_weights(fitted_model)
      %{"ar_dense_out" => %{kernel: #Nx.Tensor<f32[3][1]>}}

  """
  @spec get_ar_weights(Soothsayer.Model.t()) :: %{
          String.t() => %{optional(:bias) => Nx.Tensor.t(), kernel: Nx.Tensor.t()}
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

    * `model` - A fitted `Soothsayer.Model` struct with events or holidays
      configured. Country holidays show up under their holidefs names, for
      example `"Christmas Day_0"`.

  ## Returns

    A map of feature names to coefficient values.

  ## Examples

      iex> model = Soothsayer.new(%{events: %{"sale" => %{steps_before: 0, steps_after: 0}}})
      iex> fitted_model = Soothsayer.fit(model, data, events: events_df)
      iex> effects = Soothsayer.get_event_effects(fitted_model)
      %{"sale_0" => 45.2}

      iex> model = Soothsayer.new(%{events: %{"promo" => %{steps_before: 1, steps_after: 1}}})
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
      ...>   "trend" => Nx.template({1, 1, 11}, :f32),
      ...>   "yearly" => Nx.template({1, 1, 12}, :f32),
      ...>   "weekly" => Nx.template({1, 1, 6}, :f32),
      ...>   "daily" => Nx.template({1, 1, 12}, :f32)
      ...> }
      iex> Axon.Display.as_graph(Soothsayer.display_network(model), input)

  """
  @spec display_network(Soothsayer.Model.t()) :: Axon.t()
  def display_network(%Model{} = model) do
    Model.display_network(model.config)
  end
end
