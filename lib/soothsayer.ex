defmodule Soothsayer do
  @moduledoc """
  The main module for the Soothsayer library, providing functions for creating, fitting, and using time series forecasting models.
  """

  require Logger

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Conformal
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
        growth: :linear,
        regularization: nil
      },
      seasonality: %{
        mode: :additive,
        regularization: nil,
        yearly: %{enabled: true, fourier_terms: 6},
        weekly: %{enabled: true, fourier_terms: 3},
        daily: %{enabled: :auto, fourier_terms: 6},
        custom: %{}
      },
      frequency: :auto,
      ar: %{enabled: false, lags: 0, layers: [], regularization: nil, forecast_steps: 1},
      events: %{},
      holidays: %{
        countries: [],
        steps_before: 0,
        steps_after: 0,
        mode: :additive,
        regularization: nil,
        types: [:public],
        language: "en"
      },
      regressors: %{},
      lagged_regressors: %{},
      lagged_regressors_layers: [],
      quantiles: [],
      missing: %{impute: true, impute_linear: 10, impute_rolling: 10, drop_samples: false},
      epochs: :auto,
      learning_rate: :auto,
      recency: %{weight: 2, start: 0.0},
      schedule: :one_cycle,
      optimizer: :adam,
      batch_size: nil,
      seed: nil
    }

    merged_config =
      default_config
      |> deep_merge(config)
      |> Map.update!(:regressors, &Regressors.normalize_config!/1)

    validate_config!(merged_config)

    merged_config
    |> Map.update!(:quantiles, &Quantiles.normalize_config!/1)
    |> Map.update!(:holidays, &Holidays.normalize_config!/1)
    |> Map.update!(:events, &fill_event_defaults/1)
    |> Model.new()
  end

  @seasonality_modes [:additive, :multiplicative]

  defp validate_config!(config) do
    Seasonality.validate_config!(config.seasonality)
    validate_seasonality_mode!(config)
    validate_seasonality_enabled!(config)
    validate_regularization!(config, [:seasonality, :regularization])
    validate_regularization!(config, [:trend, :regularization])
    validate_growth!(config)
    validate_regularization!(config, [:ar, :regularization])
    Frequency.validate!(config.frequency)
    validate_lagged_regressors!(config)
    validate_lagged_regressors_layers!(config)
    validate_forecast_steps!(config)
    validate_missing!(config)
    validate_events!(config)
    validate_training_options!(config)
    validate_recency!(config)
  end

  defp validate_recency!(%{recency: %{weight: weight, start: start}})
       when (is_nil(weight) or (is_number(weight) and weight >= 1)) and
              is_number(start) and start >= 0 and start < 1,
       do: :ok

  defp validate_recency!(%{recency: recency}) do
    raise ArgumentError,
          "recency must be %{weight: nil | number >= 1, start: fraction in [0, 1)}, " <>
            "got #{inspect(recency)}"
  end

  @event_defaults %{steps_before: 0, steps_after: 0, mode: :additive}
  @event_modes [:additive, :multiplicative]

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

      validate_event_window!(name, spec)

      unless spec[:recurring] in [nil, :yearly] do
        raise ArgumentError,
              "events.#{name}.recurring must be :yearly or left out, got #{inspect(spec[:recurring])}"
      end

      unless Map.get(spec, :mode, :additive) in @event_modes do
        raise ArgumentError,
              "events.#{name}.mode must be one of #{inspect(@event_modes)}, got #{inspect(spec.mode)}"
      end

      validate_regularization!(%{events: %{name => spec}}, [:events, name, :regularization])
    end

    :ok
  end

  defp validate_events!(%{events: events}) do
    raise ArgumentError, "events must be a map of event names to windows, got #{inspect(events)}"
  end

  defp validate_event_window!(name, spec) do
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
  end

  defp validate_lagged_regressors_layers!(config) do
    layers = Map.get(config, :lagged_regressors_layers, [])

    unless is_list(layers) and Enum.all?(layers, &(is_integer(&1) and &1 > 0)) do
      raise ArgumentError,
            "lagged_regressors_layers must be a list of positive integers, got #{inspect(layers)}"
    end
  end

  defp validate_growth!(%{trend: %{growth: growth}}) do
    unless growth in [:linear, :discontinuous] do
      raise ArgumentError,
            "trend.growth must be :linear or :discontinuous, got #{inspect(growth)}. " <>
              "NeuralProphet's growth \"off\" is trend: %{enabled: false}."
    end
  end

  defp validate_regularization!(config, path) do
    value = get_in(config, path)

    unless is_nil(value) or (is_number(value) and value >= 0) do
      raise ArgumentError,
            "#{Enum.join(path, ".")} must be nil or a number >= 0, got #{inspect(value)}"
    end
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
        every year on the same month and day. An event or regressor with
        `mode: :multiplicative` scales with the trend instead of adding to
        it, and `holidays: %{mode: :multiplicative}` does the same for every
        holiday. A `regularization` on an event, a regressor, the holidays or
        the seasonality is an L1 penalty on its coefficients, like the one on
        `ar` and `trend`.

    When the model config lists `regressors`, `data` must contain a column
    for each of them. With `holidays: %{countries: [...]}` every holiday of
    those countries becomes an event of its own, named as dayoff names it,
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
    Regressors.validate_columns!(data, Regressors.names(model.config))
    Seasonality.validate_condition_columns!(data, model.config)
    Regressors.validate_columns!(data, LaggedRegressors.names(model.config))

    # Every series is prepared on its own: sorted, on the frequency grid,
    # gaps imputed. What the series share (frequency, the time axis, the
    # :auto seasonalities, holidays) is settled from all of them together.
    prepared = Enum.map(split_series(data, model.config), &prepare_series(&1, model.config))
    frequency = shared_frequency(prepared)

    all_timestamps =
      prepared |> Enum.flat_map(& &1.timestamps) |> Enum.uniq() |> Enum.sort(NaiveDateTime)

    config =
      model.config
      |> Map.put(:frequency, frequency)
      |> Map.update!(:seasonality, &Seasonality.resolve_auto(&1, all_timestamps, frequency))
      |> put_holiday_events(all_timestamps)
      |> Map.merge(Trend.changepoint_metadata(all_timestamps, model.config))

    model = %{model | config: config}
    time_span = Timestamp.days_since(List.last(all_timestamps), List.first(all_timestamps))

    # One training sample per forecast origin, per series. Without AR that
    # is every timestamp on its own. With AR a sample is the origin's lags
    # followed by its forecast_steps targets, see AR.training_samples/4,
    # and every time-based feature is gathered at all of those positions so
    # the network can evaluate the components at the lag timestamps too.
    samples = Enum.map(prepared, &training_samples(&1, model, events_df, time_span))
    x = samples |> Enum.map(& &1.x) |> concatenate_inputs()
    y_normalized = samples |> Enum.map(& &1.y) |> Nx.concatenate(axis: 0)

    {x_normalized, x_norm} = normalize_inputs(x, config)

    # The network is rebuilt now that the y normalization is known, since
    # multiplicative seasonality needs the series level as a constant.
    config =
      Map.put(config, :normalization, %{x: x_norm, y: y_normalization(samples)})

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

    # What prediction needs to look up, per series: the observed values so
    # the lag positions of a forecast can reach into the training period,
    # the regressor values there, and each series' own scale.
    training_data = %{
      series: Map.new(samples, &{&1.id, &1.entry}),
      event_dates: Events.frame_dates(events_df)
    }

    %{fitted_model | config: Map.put(config, :training_data, training_data)}
  end

  # Single series: the whole frame under the id nil.
  defp split_series(data, _config), do: [{nil, data}]

  # Sorted timestamps, missing rows and values dropped, regridded or
  # imputed, and the condition columns read from the prepared frame.
  # Whatever is still missing is NaN in `data` and listed in `unfilled`.
  defp prepare_series({id, frame}, config) do
    timestamps = Timestamp.from_series(frame["ds"])
    Timestamp.validate_sorted!(timestamps)
    frequency = resolve_frequency(config.frequency, timestamps)
    {frame, unfilled} = MissingData.prepare(frame, config, frequency)
    timestamps = Timestamp.from_series(frame["ds"])

    %{
      id: id,
      data: frame,
      timestamps: timestamps,
      unfilled: unfilled,
      frequency: frequency,
      conditions: Seasonality.condition_values(frame, config)
    }
  end

  defp shared_frequency([%{frequency: frequency}]), do: frequency

  # The inputs, targets and the training entry of one series.
  defp training_samples(
         %{id: id, data: data, timestamps: timestamps} = series,
         model,
         events_df,
         time_span
       ) do
    config = model.config

    # The mean and std come from the known values only, so the NaNs left at
    # unfilled positions stay NaN and never reach a training sample.
    y_values = data["y"] |> Series.cast({:f, 64}) |> Series.to_list()
    y_full = Nx.tensor(y_values, type: {:f, 32})
    known_y = y_values |> Enum.reject(&(&1 == :nan)) |> Nx.tensor(type: {:f, 32})
    {_known_normalized, y_mean, y_std} = normalize(Nx.new_axis(known_y, -1))
    y_full_normalized = Nx.divide(Nx.subtract(y_full, y_mean), y_std)

    {y_normalized, ar_inputs, position_indices} =
      if ar_enabled?(model) do
        max_lags = max(config.ar.lags, LaggedRegressors.max_lags(config))

        skip_positions =
          series.unfilled |> Map.values() |> Enum.reduce(MapSet.new(), &MapSet.union/2)

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

    x =
      %{"trend" => trend_input(timestamps, config)}
      |> Map.merge(seasonality_inputs(timestamps, config, series.conditions))
      |> put_events_input(model, timestamps, events_df)
      |> put_training_regressors_input(model, timestamps, data)
      |> Map.new(fn {key, features} -> {key, Nx.take(features, position_indices, axis: 0)} end)
      |> Map.merge(ar_inputs)
      |> put_sample_weight(timestamps, position_indices, config, time_span)

    y_normalized_values = Nx.to_flat_list(y_full_normalized)

    known_values =
      timestamps
      |> Enum.zip(y_normalized_values)
      |> Enum.reject(fn {_timestamp, value} -> value == :nan end)
      |> Map.new()

    entry = %{
      timestamps: timestamps,
      y_normalized: y_normalized_values,
      known_values: known_values,
      last_timestamp: Enum.max(timestamps, NaiveDateTime),
      normalization: %{mean: y_mean, std: y_std},
      regressors:
        Map.new(
          Regressors.names(config) ++ Seasonality.condition_columns(config),
          &{&1, Regressors.values_by_timestamp(data, &1)}
        ),
      lagged_regressors:
        Map.new(LaggedRegressors.names(config), &{&1, Regressors.values_by_timestamp(data, &1)})
    }

    %{id: id, x: x, y: y_normalized, entry: entry}
  end

  defp concatenate_inputs([x]), do: x

  defp concatenate_inputs(inputs) do
    Map.new(hd(inputs), fn {key, _tensor} ->
      {key, inputs |> Enum.map(& &1[key]) |> Nx.concatenate(axis: 0)}
    end)
  end

  # The level the network bakes in, see Soothsayer.Model.
  defp y_normalization([%{entry: %{normalization: normalization}}]), do: normalization

  # The trend features of some timestamps on the model's time axis: the
  # numeric time from first_timestamp and the changepoint columns.
  defp trend_input(timestamps, config) do
    t = Trend.date_to_numeric(timestamps, config.first_timestamp) |> Nx.new_axis(-1)

    changepoint_features =
      Trend.build_changepoint_features(
        t,
        config.changepoint_positions,
        Trend.basis(config),
        Trend.growth(config)
      )

    Trend.build_trend_input(t, changepoint_features)
  end

  @doc false
  # The training entry of one series: timestamps, normalized values,
  # regressor values and scale. `nil` is the id of a single series model.
  def series_entry(%Model{config: %{training_data: %{series: series}}}, id) do
    case Map.fetch(series, id) do
      {:ok, entry} ->
        entry

      :error ->
        raise ArgumentError,
              "Unknown series #{inspect(id)}, the model was fitted on #{inspect(Map.keys(series))}"
    end
  end

  def series_entry(%Model{}, _id) do
    raise ArgumentError, "Model has not been fitted yet"
  end

  defp resolve_frequency(:auto, timestamps), do: Frequency.infer(timestamps)
  defp resolve_frequency(frequency, _timestamps), do: frequency

  # The component columns of predict, in order. Custom seasonalities come
  # after the built-in ones.
  defp component_columns(config) do
    [:trend] ++
      Enum.map(Seasonality.periods(config), &Model.seasonality_key/1) ++
      [:ar, :events, :regressors, :lagged_regressors]
  end

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
      |> Enum.map(fn {quantile, tensor} -> {Quantiles.column_name(quantile), tensor} end)

    # Disabled components are scalar zeros and stay out of the frame, except
    # the trend, which carries the level of the series even when disabled
    # and is needed for the columns to add up to yhat.
    component_columns =
      for key <- component_columns(model.config),
          tensor = components[key],
          key == :trend or Nx.rank(tensor) == 2,
          do: {Atom.to_string(key), tensor}

    columns =
      [{"yhat", components.combined}] ++
        quantile_columns ++ interval_columns(model, components) ++ component_columns

    DataFrame.new([
      {"ds", x} | Enum.map(columns, fn {name, tensor} -> {name, column(tensor, rows)} end)
    ])
  end

  # Conformal bounds, once the model was calibrated with calibrate/3.
  defp interval_columns(%Model{config: %{calibration: calibration}}, components)
       when is_map(calibration) do
    {lower, upper} =
      Conformal.bounds(calibration, components.step, components.combined, components.quantiles)

    [{"yhat_lower", lower}, {"yhat_upper", upper}]
  end

  defp interval_columns(_model, _components), do: []

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
        quantiles: %{0.1 => #Nx.Tensor<...>, 0.9 => #Nx.Tensor<...>},
        step: #Nx.Tensor<...>
      }

  `:step` is how many steps past the last observation each row is, `{n, 1}`,
  1 for rows at or before it and for every row without auto-regression. With
  auto-regression a row further out than `forecast_steps` is a chained
  forecast, and its step says how far the chain went.

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
          quantiles: %{float() => Nx.Tensor.t()},
          step: Nx.Tensor.t()
        }
  def predict_components(%Model{} = model, %Series{} = x, opts \\ []) do
    events_df = Keyword.get(opts, :events)
    history = Keyword.get(opts, :history)
    regressors_df = Keyword.get(opts, :regressors)
    validate_regressors_option!(model, regressors_df)

    prediction_timestamps = Timestamp.from_series(x)
    entry = series_entry(model, nil)

    regressor_values = %{
      regressors:
        Regressors.known_values(
          entry,
          regressors_df,
          Regressors.names(model.config) ++ Seasonality.condition_columns(model.config)
        ),
      lagged_regressors: LaggedRegressors.known_values(entry, regressors_df, model.config)
    }

    {x_input, step_numbers, steps_ahead} =
      if ar_enabled?(model) do
        {observed_values, last_observed} = known_values(model, entry, history)
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

        # Steps ahead of the last observation, past forecast_steps too, so
        # chained rows are told apart from calibrated ones.
        steps_ahead =
          Enum.map(prediction_timestamps, fn timestamp ->
            max(Frequency.steps_between(last_observed, timestamp, model.config.frequency), 1)
          end)

        {inputs, step_numbers, steps_ahead}
      else
        samples = Enum.map(prediction_timestamps, &[&1])
        required = MapSet.new(prediction_timestamps)

        inputs =
          build_time_inputs(model, samples, required, events_df, regressor_values.regressors)

        ones = List.duplicate(1, length(prediction_timestamps))
        {inputs, ones, ones}
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

    components = denormalize_components(predictions, entry.normalization)

    components
    |> Map.put(
      :quantiles,
      denormalize_quantiles(
        quantile_outputs,
        model.config.quantiles,
        components.combined,
        entry.normalization
      )
    )
    |> Map.put(:step, steps_ahead |> Nx.tensor() |> Nx.reshape({:auto, 1}))
  end

  defp select_step(outputs, step_index) when is_tuple(outputs) do
    outputs |> Tuple.to_list() |> Enum.map(&select_step(&1, step_index))
  end

  # Disabled components are scalar zeros and stay that way
  defp select_step(output, step_index) do
    if Nx.rank(output) == 2, do: Nx.take_along_axis(output, step_index, axis: 1), else: output
  end

  defp denormalize_quantiles(nil, _quantiles, _combined, _normalization), do: %{}

  defp denormalize_quantiles(outputs, quantiles, combined, %{mean: mean, std: std}) do
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
      steps_after: config.holidays.steps_after,
      mode: config.holidays.mode,
      regularization: config.holidays.regularization
    }

    config
    |> Map.update!(:events, &Map.merge(&1, Map.new(names, fn name -> {name, window} end)))
    |> put_in([:holidays, :names], names)
  end

  defp seasonality_inputs(timestamps, config, conditions, opts \\ []) do
    timestamps
    |> Seasonality.build_features(config, conditions, opts)
    |> Map.new(fn {period, features} -> {Atom.to_string(period), features} end)
  end

  defp put_training_regressors_input(x, model, timestamps, data) do
    case Regressors.names(model.config) do
      [] -> x
      names -> Map.put(x, "regressors", Regressors.build_features(timestamps, data, names))
    end
  end

  # Regressor values must exist at the required timestamps (the lag
  # positions and whatever was asked for). Other target positions of a
  # sample get the training mean, which normalizes to zero, since a value
  # there only feeds that position's own output.
  defp put_regressors_input(x, model, timestamps, regressor_values, required_timestamps) do
    case Regressors.names(model.config) do
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

  defp validate_regressors_option!(%Model{} = model, regressors_df) do
    names = Regressors.names(model.config)
    conditions = Seasonality.condition_columns(model.config)

    case {names ++ conditions, regressors_df} do
      {[], _frame} ->
        :ok

      {_columns, nil} ->
        raise ArgumentError,
              "This model was fitted with regressors #{inspect(names)} and seasonality " <>
                "conditions #{inspect(conditions)}. Pass regressors: a dataframe with " <>
                "\"ds\" and those columns to predict."

      {columns, %DataFrame{} = frame} ->
        Regressors.validate_columns!(frame, columns)
    end
  end

  # Builds every input that depends only on the timestamp (trend,
  # seasonality, events and regressors) for a list of samples, each a list
  # of `positions` timestamps, as {samples, positions, features} tensors.
  # AR is added separately since it depends on previous values.
  defp build_time_inputs(model, samples, required_timestamps, events_df, regressor_values) do
    config = model.config
    timestamps = List.flatten(samples)
    positions = AR.positions(config)

    %{"trend" => trend_input(timestamps, config)}
    |> Map.merge(
      seasonality_inputs(
        timestamps,
        config,
        Map.take(regressor_values, Seasonality.condition_columns(config)),
        required: required_timestamps
      )
    )
    |> put_events_input(model, timestamps, events_df)
    |> put_regressors_input(model, timestamps, regressor_values, required_timestamps)
    |> Map.new(fn {key, features} ->
      {key, Nx.reshape(features, {length(samples), positions, :auto})}
    end)
  end

  # Observed values in normalized y space, keyed by timestamp, and the last
  # observed timestamp. Training data comes first, then `history` overrides
  # or extends it.
  defp known_values(_model, entry, nil) do
    {AR.known_values(entry), entry.last_timestamp}
  end

  # History gets the same treatment as training data with lags: put on the
  # frequency grid, trailing gap dropped, the rest imputed. Values still
  # missing stay unknown, and a forecast whose lags need them is NaN. The
  # last observed timestamp is the last one with a known value, so a
  # trailing gap is forecast like the future rather than looked up.
  defp known_values(model, entry, %DataFrame{} = history) do
    validate_history!(history)
    %{mean: mean, std: std} = entry.normalization
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

    {training_values, last_training_timestamp} = known_values(model, entry, nil)

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

  # The recency weights favour the last part of the training span. The
  # targets of a sample are the last forecast_steps of its positions, and
  # without AR the one position is the target.
  defp put_sample_weight(x, _timestamps, _position_indices, %{recency: %{weight: weight}}, _span)
       when is_nil(weight) or weight == 1,
       do: x

  defp put_sample_weight(x, timestamps, position_indices, config, time_span) do
    steps = AR.forecast_steps(config)

    target_times =
      timestamps
      |> Trend.date_to_numeric(config.first_timestamp)
      |> Nx.divide(max(time_span, 1.0e-9))
      |> Nx.take(position_indices, axis: 0)
      |> then(&Nx.slice_along_axis(&1, Nx.axis_size(&1, 1) - steps, steps, axis: 1))

    Map.put(x, "sample_weight", Trainer.recency_weights(target_times, config.recency))
  end

  # The lags are already in normalized y space, which is the space the
  # components subtracted from them predict in, so they must stay there.
  # The sample weights are loss weights, not features.
  @unnormalized_inputs ["ar", "sample_weight"]

  defp normalize_inputs(x, config) do
    Enum.reduce(x, {%{}, %{}}, fn {key, tensor}, acc ->
      normalize_single_input(key, tensor, acc, config)
    end)
  end

  defp normalize_single_input(key, tensor, {normalized, norm_params}, _config)
       when key in @unnormalized_inputs do
    {Map.put(normalized, key, tensor), norm_params}
  end

  # The trend features are scaled by the training span, not z-scored, so t
  # runs from 0 to 1 over the training data and each changepoint hinge
  # max(0, t - s) keeps those units, as in NeuralProphet. Z-scoring each
  # hinge on its own blows up the late ones (few nonzero values, tiny
  # standard deviation), which lets the slope of the last segment swing
  # with the last few days of data and then extrapolate that swing.
  defp normalize_single_input("trend" = key, tensor, {normalized, norm_params}, config) do
    features = Nx.axis_size(tensor, 2)
    time_columns = Trend.time_columns(config)
    span = tensor[[.., .., 0]] |> Nx.reduce_max()

    # Only the time columns are in days; the intercept columns of
    # discontinuous growth are 0 or 1 and stay that way.
    std =
      List.duplicate(span, time_columns) ++ List.duplicate(1.0, features - time_columns)

    std = Nx.tensor(Enum.map(std, &Nx.to_number/1), type: {:f, 32})

    norm_param = %{mean: Nx.broadcast(0.0, {features}), std: std}

    {Map.put(normalized, key, Nx.divide(tensor, std)), Map.put(norm_params, key, norm_param)}
  end

  defp normalize_single_input(key, tensor, {normalized, norm_params}, _config) do
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
  Calibrates prediction intervals on a frame the model has not seen, so
  that `predict/3` adds `yhat_lower` and `yhat_upper` columns covering a
  future point with probability `1 - alpha`. See `Soothsayer.Conformal`.

  `calibration` has `ds` and `y` for the period right after the training
  data. Options: `:alpha` (default `0.1`, or a `{lower, upper}` pair with
  `:cqr`), `:method` (`:naive`, the default, around `yhat`, or `:cqr`
  around the configured quantiles), `:events` and `:regressors` for the
  calibration dates as in `predict/3`.

  ## Examples

      iex> calibrated = Soothsayer.calibrate(fitted_model, calibration_df, alpha: 0.1)
      iex> Soothsayer.predict(calibrated, future_dates)
      #Explorer.DataFrame<
        Polars[30 x 7]
        ds date [...]
        yhat f64 [...]
        yhat_lower f64 [...]
        yhat_upper f64 [...]
        ...
      >

  """
  @spec calibrate(Soothsayer.Model.t(), Explorer.DataFrame.t(), keyword()) ::
          Soothsayer.Model.t()
  def calibrate(%Model{} = model, %DataFrame{} = calibration, opts \\ []) do
    validate_training_data!(calibration)
    put_in(model.config[:calibration], Conformal.calibrate(model, calibration, opts))
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
      configured. Country holidays show up under their dayoff names, for
      example `"Christmas Day_0"`.

  ## Returns

    A map of feature names to coefficient values.

  ## Examples

      iex> model = Soothsayer.new(%{events: %{"sale" => %{steps_before: 0, steps_after: 0}}})
      iex> fitted_model = Soothsayer.fit(model, data, events: events_df)
      iex> effects = Soothsayer.get_event_effects(fitted_model)
      %{"sale_0" => 0.9}

      iex> model = Soothsayer.new(%{events: %{"promo" => %{steps_before: 1, steps_after: 1}}})
      iex> fitted_model = Soothsayer.fit(model, data, events: events_df)
      iex> effects = Soothsayer.get_event_effects(fitted_model)
      %{"promo_-1" => 0.2, "promo_0" => 1.1, "promo_+1" => 0.1}

  Coefficients are per normalized unit of the event feature: for an
  additive event the change in normalized y, for a `mode: :multiplicative`
  event the fraction of the trend.

  """
  @spec get_event_effects(Soothsayer.Model.t()) :: %{String.t() => float()}
  def get_event_effects(%Model{} = model) do
    Events.get_effects(model)
  end

  @doc """
  Extracts the learned future regressor coefficients from a fitted model.

  Coefficients are in normalized units: the change in normalized y for a one
  standard deviation change in the regressor, or for a `mode: :multiplicative`
  regressor the fraction of the trend per standard deviation. Positive means
  the regressor pushes the forecast up. A regressor with `layers` has no
  single coefficient, it maps to its network's weights by layer name.

  ## Examples

      iex> model = Soothsayer.new(%{regressors: ["temperature"]})
      iex> fitted_model = Soothsayer.fit(model, data)
      iex> Soothsayer.get_regressor_effects(fitted_model)
      %{"temperature" => 0.42}

      iex> model = Soothsayer.new(%{regressors: %{"temperature" => %{layers: [8]}}})
      iex> fitted_model = Soothsayer.fit(model, data)
      iex> Soothsayer.get_regressor_effects(fitted_model)
      %{"temperature" => %{"regressor_temperature_dense_0" => %{kernel: ..., bias: ...},
                           "regressor_temperature_dense_out" => %{kernel: ...}}}

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
