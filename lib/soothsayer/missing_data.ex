defmodule Soothsayer.MissingData do
  @moduledoc """
  Missing values and missing rows in the training data, handled the way
  NeuralProphet does.

  A value is missing when it is `nil` or NaN. Without auto-regression the
  rows with a missing `y` are dropped, since every other component only
  looks at timestamps. With auto-regression the lags are the previous
  rows, so the data is first put on the model's frequency grid (absent
  timestamps become rows of missing values), trailing rows with a missing
  `y` are dropped, and the remaining gaps are imputed with `impute/3`:
  short ones linearly, longer ones with a rolling mean. Whatever is still
  missing afterwards either drops the training samples that touch it
  (`missing: %{drop_samples: true}`) or raises.

  Regressor and lagged regressor columns are imputed the same way, with or
  without auto-regression.
  """

  require Logger

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Frequency
  alias Soothsayer.LaggedRegressors
  alias Soothsayer.Regressors
  alias Soothsayer.Timestamp

  @type missing_positions :: %{String.t() => MapSet.t(non_neg_integer())}

  @doc """
  Puts the training data in shape for fitting.

  Returns the rebuilt dataframe (`ds`, `y` and the configured regressor
  columns, other columns are dropped) with `:nan` where a value is still
  missing, and for every value column the set of row indices still
  missing (empty when everything was filled).

  Raises `ArgumentError` when `y` has no values at all, when a timestamp
  is not on the frequency grid (auto-regression only), and, without
  auto-regression, when regressor values stay missing and
  `missing.drop_samples` is false.
  """
  @spec prepare(DataFrame.t(), map(), Frequency.t()) :: {DataFrame.t(), missing_positions()}
  def prepare(%DataFrame{} = data, config, frequency) do
    columns = Enum.uniq(["y" | Regressors.names(config) ++ LaggedRegressors.names(config)])
    timestamps = Series.to_list(data["ds"])
    values = Map.new(columns, &{&1, column_values(data, &1)})

    if Enum.all?(values["y"], &is_nil/1) do
      raise ArgumentError, "The y column has no values, every row is nil or NaN."
    end

    with_lags? = AR.lags(config) > 0

    {timestamps, values} =
      if with_lags? do
        {timestamps, values} = regrid(timestamps, values, frequency)
        drop_trailing_missing_targets(timestamps, values)
      else
        drop_rows_with_missing_targets(timestamps, values)
      end

    imputed_columns = if with_lags?, do: columns, else: columns -- ["y"]
    values = impute_columns(values, imputed_columns, config.missing)
    remaining = Map.new(values, fn {name, list} -> {name, missing_positions(list)} end)

    {timestamps, values, remaining} =
      if with_lags?,
        do: {timestamps, values, remaining},
        else: drop_rows_with_missing_regressors(timestamps, values, remaining, config.missing)

    frame =
      values
      |> Map.new(fn {name, list} -> {name, Enum.map(list, &(&1 || :nan))} end)
      |> Map.put("ds", timestamps)
      |> DataFrame.new()

    {frame, remaining}
  end

  @doc """
  Puts the history passed to `Soothsayer.predict/3` on the frequency grid,
  drops the rows at the end whose `y` is missing (those are forecast like
  the future instead) and imputes the rest like training data. Values
  still missing come back as `nil`.
  """
  @spec fill_history(list(Timestamp.t()), list(number() | :nan | nil), Frequency.t(), map()) ::
          {list(Timestamp.t()), list(float() | nil)}
  def fill_history([], values, _frequency, _missing_config), do: {[], values}

  def fill_history(timestamps, values, frequency, missing_config) do
    values =
      Enum.map(values, fn
        :nan -> nil
        value -> value
      end)

    {timestamps, columns} = regrid(timestamps, %{"y" => values}, frequency)
    {timestamps, columns} = drop_trailing_missing_targets(timestamps, columns)
    columns = impute_columns(columns, ["y"], missing_config)
    {timestamps, columns["y"]}
  end

  @doc """
  Fills gaps in a list of values, `nil` meaning missing, the way
  NeuralProphet's `fill_linear_then_rolling_avg` does.

  First every run of missing values is interpolated linearly between its
  neighbours, up to `linear` values from each end of the run (so runs of
  at most `2 * linear` are filled completely). A run at the start or the
  end of the list has only one neighbour and is filled with that value.
  Then, in a single pass over the result, each value still missing is
  replaced by the mean of the known values in a centered window of
  `rolling + 2 * linear` positions, when at least `2 * linear` of them are
  known. Gaps longer than `2 * linear + rolling` keep their middle.

  ## Examples

      iex> Soothsayer.MissingData.impute([1.0, nil, nil, 4.0], 10, 10)
      [1.0, 2.0, 3.0, 4.0]

      iex> Soothsayer.MissingData.impute([1.0, nil, nil, nil, nil, nil, 7.0], 1, 0)
      [1.0, 2.0, nil, nil, nil, 6.0, 7.0]

      iex> Soothsayer.MissingData.impute([nil, nil, 3.0, 4.0], 1, 0)
      [nil, 3.0, 3.0, 4.0]

  """
  @spec impute(list(number() | nil), non_neg_integer(), non_neg_integer()) ::
          list(float() | nil)
  def impute(values, linear, rolling) do
    values
    |> Enum.map(fn
      nil -> nil
      value -> value / 1
    end)
    |> linear_pass(linear)
    |> rolling_pass(linear, rolling)
  end

  @doc """
  The gap between `2 * linear + rolling` and above that imputation leaves
  open, for messages.
  """
  @spec longest_filled_gap(map()) :: non_neg_integer()
  def longest_filled_gap(%{impute_linear: linear, impute_rolling: rolling}) do
    2 * linear + rolling
  end

  @doc """
  Raises the error for training samples that still touch missing values.
  """
  @spec raise_unfilled!(non_neg_integer(), map()) :: no_return()
  def raise_unfilled!(count, missing_config) do
    raise ArgumentError,
          "#{count} training samples touch missing values that couldn't be imputed " <>
            "(gaps longer than #{longest_filled_gap(missing_config)} steps). " <>
            "Set missing: %{drop_samples: true} to skip those samples, " <>
            "or fill the gaps before fitting."
  end

  # Reading

  defp column_values(data, name) do
    data[name]
    |> Series.cast({:f, 64})
    |> Series.to_list()
    |> Enum.map(fn
      :nan -> nil
      value -> value
    end)
  end

  # Regridding

  defp regrid(timestamps, values, frequency) do
    first = hd(timestamps)
    grid = [first | Frequency.range(first, List.last(timestamps), frequency)]
    on_grid = MapSet.new(grid)

    case Enum.find(timestamps, &(not MapSet.member?(on_grid, &1))) do
      nil -> :ok
      timestamp -> raise_off_grid!(timestamp, frequency)
    end

    row_by_timestamp = timestamps |> Enum.with_index() |> Map.new()
    rows = Enum.map(grid, &Map.get(row_by_timestamp, &1))
    added = length(grid) - length(timestamps)

    if added > 0 do
      Logger.info(
        "Added #{added} missing timestamps to the #{Frequency.describe(frequency)} grid"
      )
    end

    values =
      Map.new(values, fn {name, list} ->
        tuple = List.to_tuple(list)

        {name,
         Enum.map(rows, fn
           nil -> nil
           row -> elem(tuple, row)
         end)}
      end)

    {grid, values}
  end

  defp raise_off_grid!(timestamp, frequency) do
    raise ArgumentError,
          "#{Timestamp.format(timestamp)} is not on the #{Frequency.describe(frequency)} grid " <>
            "of the training data. Auto-regression needs every row a whole number of steps " <>
            "from the first one."
  end

  # Dropping rows

  defp drop_trailing_missing_targets(timestamps, values) do
    trailing = values["y"] |> Enum.reverse() |> Enum.take_while(&is_nil/1) |> length()

    if trailing > 0 do
      Logger.info("Dropped #{trailing} rows at the end with missing y values")
      keep = length(timestamps) - trailing

      {Enum.take(timestamps, keep),
       Map.new(values, fn {name, list} -> {name, Enum.take(list, keep)} end)}
    else
      {timestamps, values}
    end
  end

  defp drop_rows_with_missing_targets(timestamps, values) do
    missing = missing_positions(values["y"])

    if MapSet.size(missing) > 0 do
      Logger.info("Dropped #{MapSet.size(missing)} rows with missing y values")
      drop_rows(timestamps, values, missing)
    else
      {timestamps, values}
    end
  end

  # Without auto-regression every row is its own training sample, so a
  # regressor value still missing decides the fate of the whole row.
  defp drop_rows_with_missing_regressors(timestamps, values, remaining, missing_config) do
    rows = remaining |> Map.values() |> Enum.reduce(MapSet.new(), &MapSet.union/2)

    cond do
      MapSet.size(rows) == 0 ->
        {timestamps, values, remaining}

      missing_config.drop_samples ->
        Logger.info("Dropped #{MapSet.size(rows)} rows with missing regressor values")
        {timestamps, values} = drop_rows(timestamps, values, rows)
        {timestamps, values, Map.new(remaining, fn {name, _} -> {name, MapSet.new()} end)}

      true ->
        raise_unfilled!(MapSet.size(rows), missing_config)
    end
  end

  defp drop_rows(timestamps, values, rows) do
    keep = fn list ->
      list
      |> Enum.with_index()
      |> Enum.reject(fn {_, index} -> MapSet.member?(rows, index) end)
      |> Enum.map(&elem(&1, 0))
    end

    {keep.(timestamps), Map.new(values, fn {name, list} -> {name, keep.(list)} end)}
  end

  defp missing_positions(list) do
    for {nil, index} <- Enum.with_index(list), into: MapSet.new(), do: index
  end

  # Imputation

  defp impute_columns(values, _columns, %{impute: false}), do: values

  defp impute_columns(values, columns, missing_config) do
    Enum.reduce(columns, values, fn name, values ->
      Map.update!(values, name, &impute_column(&1, name, missing_config))
    end)
  end

  defp impute_column(list, name, missing_config) do
    missing_before = Enum.count(list, &is_nil/1)

    if missing_before == 0 do
      list
    else
      filled = impute(list, missing_config.impute_linear, missing_config.impute_rolling)
      remaining = Enum.count(filled, &is_nil/1)
      Logger.info("Imputed #{missing_before - remaining} missing values in #{name}")

      if remaining > 0 do
        Logger.warning(
          "#{remaining} missing values remain in #{name} after imputation, gaps longer " <>
            "than #{longest_filled_gap(missing_config)} steps aren't filled"
        )
      end

      filled
    end
  end

  # Every run of nils becomes {start, stop} (inclusive positions).
  defp missing_runs(values) do
    values
    |> Enum.with_index()
    |> Enum.chunk_by(fn {value, _} -> is_nil(value) end)
    |> Enum.filter(fn [{value, _} | _] -> is_nil(value) end)
    |> Enum.map(fn run -> {elem(hd(run), 1), elem(List.last(run), 1)} end)
  end

  defp linear_pass(values, 0), do: values

  defp linear_pass(values, linear) do
    tuple = List.to_tuple(values)
    last = tuple_size(tuple) - 1

    fills =
      values
      |> missing_runs()
      |> Enum.flat_map(fn {start, stop} ->
        previous = if start > 0, do: {start - 1, elem(tuple, start - 1)}
        next = if stop < last, do: {stop + 1, elem(tuple, stop + 1)}

        # Only the end of a run next to a known value gets filled, so a run
        # at the start or end of the list is filled from its one neighbour.
        from_left = if previous, do: Enum.take(start..stop, linear), else: []
        from_right = if next, do: Enum.take(start..stop, -linear), else: []

        Enum.map(Enum.uniq(from_left ++ from_right), &{&1, interpolate(&1, previous, next)})
      end)
      |> Enum.reject(fn {_, value} -> is_nil(value) end)
      |> Map.new()

    values |> Enum.with_index() |> Enum.map(fn {value, index} -> Map.get(fills, index, value) end)
  end

  defp interpolate(_position, nil, nil), do: nil
  defp interpolate(_position, nil, {_, next_value}), do: next_value
  defp interpolate(_position, {_, previous_value}, nil), do: previous_value

  defp interpolate(position, {previous_index, previous_value}, {next_index, next_value}) do
    fraction = (position - previous_index) / (next_index - previous_index)
    previous_value + fraction * (next_value - previous_value)
  end

  # A single pass over the linearly filled values: the window means are
  # taken from those, never from values this pass filled.
  defp rolling_pass(values, linear, rolling) do
    width = rolling + 2 * linear
    min_known = max(2 * linear, 1)
    tuple = List.to_tuple(values)
    last = tuple_size(tuple) - 1

    values
    |> Enum.with_index()
    |> Enum.map(fn
      {nil, index} ->
        known =
          max(index - div(width, 2), 0)..min(index + div(width - 1, 2), last)//1
          |> Enum.map(&elem(tuple, &1))
          |> Enum.reject(&is_nil/1)

        if length(known) >= min_known, do: Enum.sum(known) / length(known), else: nil

      {value, _} ->
        value
    end)
  end
end
