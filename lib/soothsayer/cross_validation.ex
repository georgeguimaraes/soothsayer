defmodule Soothsayer.CrossValidation do
  @moduledoc """
  Evaluation over several cutoffs through the history, Prophet's
  `cross_validation`.

  A cutoff is a point in the data. The model is fitted on everything up to
  it and forecasts the `horizon` rows after it, the way it would have been
  used at the time. Cutoffs are spaced `period` rows apart, the last one
  `horizon` rows before the end, going back as long as at least `initial`
  rows remain for training. The forecasts of every cutoff are stacked with
  their step ahead, so the metrics say how a configuration behaves across
  the history and by horizon, not only at the end of the series.

  `Soothsayer.backtest/3` is the other protocol: one cutoff, every held
  out row forecast from every origin before it. This one refits per cutoff
  and forecasts each cutoff's horizon once, from the cutoff.

  Rows, not durations: `horizon`, `period` and `initial` count observations
  of a series, like `ar.forecast_steps`. With several series a cutoff is
  a timestamp shared by all of them, chosen on the union of their dates,
  and each series contributes its own rows up to and after it.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Backtest
  alias Soothsayer.Timestamp

  @type result :: %{
          cutoffs: list(Date.t() | NaiveDateTime.t()),
          metrics: Backtest.metrics(),
          by_step: %{pos_integer() => Backtest.metrics()},
          predictions: DataFrame.t()
        }

  @doc """
  Runs the cross-validation.

  ## Options

    * `:horizon` - Rows to forecast after each cutoff. Defaults to the
      model's `ar.forecast_steps`, or 1 without auto-regression.
    * `:period` - Rows between cutoffs. Defaults to half the horizon.
    * `:initial` - Rows the first cutoff must leave for training. Defaults
      to three times the horizon.
    * `:cutoffs` - The cutoff dates themselves, which replaces the three
      above. Each must leave at least two training rows before it and one
      row after it.
    * `:events` - Events dataframe, passed to every fit and predict.
    * `:regressors` - Regressors dataframe for prediction. Defaults to the
      data itself, which holds the regressor columns for the forecast dates.

  ## Returns

    A map with the `:cutoffs` used, overall `:metrics`, `:by_step` metrics
    keyed by step ahead, and a `:predictions` dataframe with columns
    `cutoff`, `ds`, `step`, `y` and `yhat`, the quantile columns when
    quantiles are configured, `yhat_lower` and `yhat_upper` when the model
    is calibrated, and the id column first for several series. Rows whose
    `y` is missing are not scored.
  """
  @spec run(Soothsayer.Model.t(), DataFrame.t(), keyword()) :: result()
  def run(%Soothsayer.Model{} = model, %DataFrame{} = data, opts \\ []) do
    horizon = Keyword.get(opts, :horizon, AR.forecast_steps(model.config))
    period = Keyword.get(opts, :period, max(div(horizon, 2), 1))
    initial = Keyword.get(opts, :initial, 3 * horizon)
    events = Keyword.get(opts, :events)
    regressors = Keyword.get(opts, :regressors, data)
    column = Soothsayer.Series.column(model.config)

    timestamps = sorted_timestamps(data)

    cutoffs =
      case Keyword.get(opts, :cutoffs) do
        nil -> cutoffs(timestamps, horizon, period, initial)
        given -> validate_cutoffs!(given, timestamps)
      end

    frames =
      Enum.map(cutoffs, fn cutoff ->
        {train, test} = split_at(data, cutoff, horizon, column)
        fitted = Soothsayer.fit(model, train, fit_options(events))
        forecast(fitted, cutoff, test, column, events, regressors)
      end)

    predictions = DataFrame.concat_rows(frames)

    %{
      cutoffs: cutoffs,
      metrics: Backtest.evaluate(predictions),
      by_step: Backtest.metrics_by_step(predictions),
      predictions: predictions
    }
  end

  @doc """
  The cutoffs for `timestamps` (sorted, unique): the last one `horizon`
  rows before the end, then every `period` rows back while `initial` rows
  remain before it, oldest first.

  ## Examples

      iex> dates = Enum.map(0..19, &Date.add(~D[2024-01-01], &1))
      iex> Soothsayer.CrossValidation.cutoffs(dates, 3, 5, 6)
      [~D[2024-01-07], ~D[2024-01-12], ~D[2024-01-17]]

  """
  @spec cutoffs(list(Timestamp.input()), pos_integer(), pos_integer(), pos_integer()) ::
          list(Timestamp.input())
  def cutoffs(timestamps, horizon, period, initial)
      when horizon > 0 and period > 0 and initial > 0 do
    last_index = length(timestamps) - horizon - 1

    if last_index < initial - 1 do
      raise ArgumentError,
            "Cross-validation needs at least initial + horizon rows " <>
              "(#{initial} + #{horizon}), got #{length(timestamps)}"
    end

    last_index
    |> Stream.iterate(&(&1 - period))
    |> Enum.take_while(&(&1 >= initial - 1))
    |> Enum.reverse()
    |> Enum.map(&Enum.at(timestamps, &1))
  end

  defp validate_cutoffs!([_ | _] = cutoffs, timestamps) do
    first = Enum.at(timestamps, 1)
    last = List.last(timestamps)

    for cutoff <- cutoffs,
        Timestamp.days_since(cutoff, first) < 0 or Timestamp.days_since(last, cutoff) <= 0 do
      raise ArgumentError,
            "Cutoff #{inspect(cutoff)} leaves no training rows before it or no rows after " <>
              "it, the data runs from #{inspect(List.first(timestamps))} to #{inspect(last)}"
    end

    Enum.sort_by(cutoffs, &Timestamp.to_naive_datetime/1, NaiveDateTime)
  end

  defp validate_cutoffs!(other, _timestamps) do
    raise ArgumentError, "cutoffs must be a non-empty list of dates, got #{inspect(other)}"
  end

  # Sorted unique timestamps over every series, in the frame's own kind
  defp sorted_timestamps(data) do
    data["ds"]
    |> Series.to_list()
    |> Enum.uniq()
    |> Enum.sort_by(&Timestamp.to_naive_datetime/1, NaiveDateTime)
  end

  # Training rows up to the cutoff and the next `horizon` rows after it,
  # per series when there are several.
  defp split_at(data, cutoff, horizon, nil) do
    {before, after_cutoff} = partition(data, cutoff)
    {before, DataFrame.head(after_cutoff, horizon)}
  end

  defp split_at(data, cutoff, horizon, column) do
    {trains, tests} =
      data
      |> Soothsayer.Series.unique_ids!(column)
      |> Enum.map(fn id ->
        split_at(Soothsayer.Series.rows_of(data, column, id), cutoff, horizon, nil)
      end)
      |> Enum.unzip()

    {DataFrame.concat_rows(trains), DataFrame.concat_rows(tests)}
  end

  defp partition(data, cutoff) do
    cutoff_naive = Timestamp.to_naive_datetime(cutoff)

    mask =
      data["ds"]
      |> Timestamp.from_series()
      |> Enum.map(&(NaiveDateTime.compare(&1, cutoff_naive) != :gt))

    mask_series = Series.from_list(mask)
    {DataFrame.mask(data, mask_series), DataFrame.mask(data, Series.not(mask_series))}
  end

  defp forecast(fitted, cutoff, test, column, events, regressors) do
    input = if column, do: DataFrame.select(test, ["ds", column]), else: test["ds"]
    options = [regressors: regressors] |> maybe_put(:events, events)
    predictions = Soothsayer.predict(fitted, input, options)

    steps =
      if column do
        test[column]
        |> Series.to_list()
        |> Enum.chunk_by(& &1)
        |> Enum.flat_map(&Enum.to_list(1..length(&1)))
      else
        Enum.to_list(1..DataFrame.n_rows(test))
      end

    extra =
      predictions
      |> DataFrame.names()
      |> Enum.filter(&String.starts_with?(&1, "yhat_"))

    predictions
    |> DataFrame.select(["ds"] ++ List.wrap(column) ++ ["yhat"] ++ extra)
    |> DataFrame.put("y", test["y"])
    |> DataFrame.put("step", Series.from_list(steps))
    |> DataFrame.put("cutoff", Series.from_list(List.duplicate(cutoff, DataFrame.n_rows(test))))
    |> DataFrame.select(List.wrap(column) ++ ["cutoff", "ds", "step", "y", "yhat"] ++ extra)
    |> DataFrame.filter_with(&Series.not(Series.is_nil(&1["y"])))
  end

  defp fit_options(nil), do: []
  defp fit_options(events), do: [events: events]

  defp maybe_put(options, _key, nil), do: options
  defp maybe_put(options, key, value), do: Keyword.put(options, key, value)
end
