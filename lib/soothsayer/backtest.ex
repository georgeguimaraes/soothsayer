defmodule Soothsayer.Backtest do
  @moduledoc """
  Rolling-origin evaluation of a model configuration on held out data.

  Splits the data into a training and a validation part, fits the model on
  the training part, then walks through the validation part one origin at a
  time: from each origin it forecasts the next `horizon` dates using only
  what was observed up to that origin, and compares against the actual
  values. Every validation date is therefore forecast once from each of the
  `horizon` steps before it, which is the same protocol NeuralProphet uses
  for its validation metrics.

  For models without auto-regression the forecast for a date doesn't depend
  on the origin, so the per-step metrics come out identical.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Conformal
  alias Soothsayer.Quantiles

  @type metrics :: %{
          optional(:coverage) => float(),
          optional(:mean_interval_width) => float(),
          mean_absolute_error: float(),
          root_mean_squared_error: float()
        }

  @type result :: %{
          model: Soothsayer.Model.t(),
          metrics: metrics(),
          by_step: %{pos_integer() => metrics()},
          predictions: DataFrame.t()
        }

  @doc """
  Runs the backtest.

  ## Options

    * `:validation_fraction` - Share of rows held out at the end, default `0.1`.
      Like NeuralProphet's `split_df`, the validation part is the last
      `max(1, trunc(rows * fraction))` rows.
    * `:horizon` - Steps ahead to forecast from each origin. Defaults to the
      model's `ar.forecast_steps`, or 1 without auto-regression.
    * `:events` - Events dataframe, passed to both fit and predict.
    * `:regressors` - Regressors dataframe for prediction. Defaults to the
      validation part itself, which holds the regressor columns for those dates.

  ## Returns

    A map with the fitted `:model`, overall `:metrics`, `:by_step` metrics
    keyed by step ahead, and a `:predictions` dataframe with columns
    `origin`, `ds`, `step`, `y` and `yhat`, plus the quantile columns when
    quantiles are configured and `yhat_lower`/`yhat_upper` when the model
    was calibrated with `Soothsayer.calibrate/3`. With an interval the
    metrics also carry its `:coverage` (share of `y` inside it) and
    `:mean_interval_width`. Validation dates whose `y` is
    missing (nil or NaN) can't be scored and are left out of both; they
    still flow into the history each forecast is made from, where they are
    imputed like training data (see `Soothsayer.MissingData`). Forecasts
    that come out NaN, because their lags reach into a gap that couldn't be
    imputed, are left out the same way.

  """
  @spec run(Soothsayer.Model.t(), DataFrame.t(), keyword()) :: result()
  def run(%Soothsayer.Model{} = model, %DataFrame{} = data, opts \\ []) do
    validation_fraction = Keyword.get(opts, :validation_fraction, 0.1)
    horizon = Keyword.get(opts, :horizon, AR.forecast_steps(model.config))
    events = Keyword.get(opts, :events)

    {train, validation} = split(data, validation_fraction, Soothsayer.Series.column(model.config))
    regressors = Keyword.get(opts, :regressors, validation)

    fitted_model = Soothsayer.fit(model, train, fit_options(events))

    predictions =
      rolling_predictions(fitted_model, validation,
        horizon: horizon,
        events: events,
        regressors: regressors
      )

    %{
      model: fitted_model,
      metrics: frame_metrics(predictions),
      by_step: metrics_by_step(predictions),
      predictions: predictions
    }
  end

  @doc """
  Forecasts every date of `validation` from every origin before it, as
  `run/3` does after fitting: a frame with `origin`, `ds`, `step`, `y`,
  `yhat`, the quantile columns and, for a calibrated model, `yhat_lower`
  and `yhat_upper`. The model must be fitted on the data right before
  `validation`, since its training data seeds the first origin.

  Options: `:horizon` (default `ar.forecast_steps`), `:events`,
  `:regressors` (default `validation`).
  """
  @spec rolling_predictions(Soothsayer.Model.t(), DataFrame.t(), keyword()) :: DataFrame.t()
  def rolling_predictions(
        %Soothsayer.Model{} = fitted_model,
        %DataFrame{} = validation,
        opts \\ []
      ) do
    case Soothsayer.Series.column(fitted_model.config) do
      nil ->
        series_rolling_predictions(fitted_model, nil, validation, opts)

      column ->
        # Every series walks its own validation rows; the frames are stacked
        # with the id column first.
        validation
        |> Soothsayer.Series.unique_ids!(column)
        |> Enum.map(&one_series_rolling_predictions(fitted_model, column, &1, validation, opts))
        |> DataFrame.concat_rows()
    end
  end

  defp one_series_rolling_predictions(fitted_model, column, id, validation, opts) do
    rows = Soothsayer.Series.rows_of(validation, column, id)

    opts =
      Keyword.update(opts, :regressors, rows, &(&1 && Soothsayer.Series.rows_of(&1, column, id)))

    predictions = series_rolling_predictions(fitted_model, id, rows, opts)
    ids = Series.from_list(List.duplicate(id, DataFrame.n_rows(predictions)))

    predictions
    |> DataFrame.put(column, ids)
    |> then(&DataFrame.select(&1, [column | DataFrame.names(&1) -- [column]]))
  end

  defp series_rolling_predictions(fitted_model, id, validation, opts) do
    horizon = Keyword.get(opts, :horizon, AR.forecast_steps(fitted_model.config))
    events = Keyword.get(opts, :events)
    regressors = Keyword.get(opts, :regressors, validation)
    validation_dates = Series.to_list(validation["ds"])
    actual_by_date = Enum.zip(validation_dates, Series.to_list(validation["y"])) |> Map.new()

    last_training_date =
      same_kind(Soothsayer.series_entry(fitted_model, id).last_timestamp, hd(validation_dates))

    predict_options = [regressors: regressors] |> maybe_put(:events, events)

    origins =
      if AR.lags(fitted_model.config) > 0 do
        ar_origins(validation, last_training_date, horizon, predict_options)
      else
        # Without lags a forecast doesn't depend on its origin: one call.
        [{last_training_date, validation_dates, predict_options}]
      end

    rows =
      Enum.flat_map(origins, fn {origin_date, target_dates, options} ->
        forecast_rows(fitted_model, id, origin_date, target_dates, options, actual_by_date)
      end)

    columns =
      ["origin", "ds", "step", "y", "yhat"] ++
        Enum.map(fitted_model.config.quantiles, &Quantiles.column_name/1) ++
        if(calibrated?(fitted_model), do: ["yhat_lower", "yhat_upper"], else: [])

    DataFrame.new(Enum.map(columns, fn column -> {column, Enum.map(rows, & &1[column])} end))
  end

  # One origin per validation row, the rows before it as history.
  defp ar_origins(validation, last_training_date, horizon, predict_options) do
    validation_dates = Series.to_list(validation["ds"])

    Enum.map(0..(length(validation_dates) - 1), fn origin_rows ->
      origin_date =
        if origin_rows == 0,
          do: last_training_date,
          else: Enum.at(validation_dates, origin_rows - 1)

      history = if origin_rows > 0, do: DataFrame.head(validation, origin_rows)

      {origin_date, Enum.slice(validation_dates, origin_rows, horizon),
       maybe_put(predict_options, :history, history)}
    end)
  end

  defp forecast_rows(fitted_model, id, origin_date, target_dates, options, actual_by_date) do
    input = prediction_input(target_dates, id, fitted_model)
    components = Soothsayer.predict_components(fitted_model, input, options)

    interval =
      if calibrated?(fitted_model) do
        {lower, upper} =
          Conformal.bounds(
            fitted_model.config.calibration,
            components.step,
            components.combined,
            components.quantiles
          )

        %{"yhat_lower" => Nx.to_flat_list(lower), "yhat_upper" => Nx.to_flat_list(upper)}
      else
        %{}
      end

    quantile_values =
      Map.new(components.quantiles, fn {quantile, tensor} ->
        {Quantiles.column_name(quantile), Nx.to_flat_list(tensor)}
      end)

    extra_columns = Map.merge(quantile_values, interval)

    target_dates
    |> Enum.zip(Nx.to_flat_list(components.combined))
    |> Enum.with_index()
    |> Enum.reject(fn {{date, yhat}, _index} ->
      missing?(actual_by_date[date]) or missing?(yhat)
    end)
    |> Enum.map(fn {{date, yhat}, index} ->
      base = %{
        "origin" => origin_date,
        "ds" => date,
        "step" => components.step |> Nx.to_flat_list() |> Enum.at(index),
        "y" => actual_by_date[date],
        "yhat" => yhat
      }

      Enum.reduce(extra_columns, base, fn {column, values}, row ->
        Map.put(row, column, Enum.at(values, index))
      end)
    end)
  end

  defp calibrated?(model), do: is_map(model.config[:calibration])

  # A series of dates for a single series model, a frame with the id column
  # for several series.
  defp prediction_input(dates, nil, _model), do: Series.from_list(dates)

  defp prediction_input(dates, id, model) do
    column = Soothsayer.Series.column(model.config)
    DataFrame.new([{"ds", dates}, {column, List.duplicate(id, length(dates))}])
  end

  # The training data keeps naive datetimes, the origin column follows the
  # validation frame's own kind.
  defp same_kind(%NaiveDateTime{} = timestamp, %Date{}), do: NaiveDateTime.to_date(timestamp)
  defp same_kind(timestamp, _like), do: timestamp

  @doc """
  Splits a dataframe into training and validation parts, validation being
  the last `max(1, trunc(rows * fraction))` rows. With a series `column`
  every series is split that way on its own.
  """
  @spec split(DataFrame.t(), float(), String.t() | nil) :: {DataFrame.t(), DataFrame.t()}
  def split(data, validation_fraction, column \\ nil)

  def split(%DataFrame{} = data, validation_fraction, nil) do
    rows = DataFrame.n_rows(data)
    validation_rows = max(1, trunc(rows * validation_fraction))
    {DataFrame.head(data, rows - validation_rows), DataFrame.tail(data, validation_rows)}
  end

  def split(%DataFrame{} = data, validation_fraction, column) do
    {trains, validations} =
      data
      |> Soothsayer.Series.unique_ids!(column)
      |> Enum.map(&split(Soothsayer.Series.rows_of(data, column, &1), validation_fraction, nil))
      |> Enum.unzip()

    {DataFrame.concat_rows(trains), DataFrame.concat_rows(validations)}
  end

  @doc """
  Mean absolute error and root mean squared error between two series.
  """
  @spec metrics(Series.t(), Series.t()) :: %{
          mean_absolute_error: float(),
          root_mean_squared_error: float()
        }
  def metrics(actual, predicted) do
    actual = actual |> Series.to_tensor() |> Nx.as_type({:f, 32})
    predicted = predicted |> Series.to_tensor() |> Nx.as_type({:f, 32})

    %{
      mean_absolute_error: Axon.Metrics.mean_absolute_error(actual, predicted) |> Nx.to_number(),
      root_mean_squared_error:
        Axon.Losses.mean_squared_error(actual, predicted, reduction: :mean)
        |> Nx.sqrt()
        |> Nx.to_number()
    }
  end

  @doc """
  Coverage (share of `actual` inside `[lower, upper]`) and mean interval
  width of a prediction interval.
  """
  @spec interval_metrics(Series.t(), Series.t(), Series.t()) :: %{
          coverage: float(),
          mean_interval_width: float()
        }
  def interval_metrics(actual, lower, upper) do
    inside =
      Series.and(
        Series.greater_equal(actual, lower),
        Series.less_equal(actual, upper)
      )

    %{
      coverage: Series.mean(Series.cast(inside, {:s, 8})),
      mean_interval_width: Series.mean(Series.subtract(upper, lower))
    }
  end

  # MAE and RMSE, plus the interval metrics when the frame has an interval:
  # the conformal columns first, else the outermost quantile columns.
  defp frame_metrics(predictions) do
    base = metrics(predictions["y"], predictions["yhat"])

    case interval_columns(predictions) do
      {lower, upper} -> Map.merge(base, interval_metrics(predictions["y"], lower, upper))
      nil -> base
    end
  end

  defp interval_columns(predictions) do
    names = DataFrame.names(predictions)

    quantile_names =
      names
      |> Enum.filter(&String.starts_with?(&1, "yhat_"))
      |> Enum.reject(&(&1 in ["yhat_lower", "yhat_upper"]))

    cond do
      "yhat_lower" in names ->
        {predictions["yhat_lower"], predictions["yhat_upper"]}

      length(quantile_names) >= 2 ->
        {predictions[List.first(quantile_names)], predictions[List.last(quantile_names)]}

      true ->
        nil
    end
  end

  defp missing?(value), do: is_nil(value) or value == :nan

  defp fit_options(nil), do: []
  defp fit_options(events), do: [events: events]

  defp metrics_by_step(predictions) do
    predictions["step"]
    |> Series.distinct()
    |> Series.to_list()
    |> Map.new(fn step ->
      rows = DataFrame.filter_with(predictions, &Series.equal(&1["step"], step))
      {step, frame_metrics(rows)}
    end)
  end

  defp maybe_put(options, _key, nil), do: options
  defp maybe_put(options, key, value), do: Keyword.put(options, key, value)
end
