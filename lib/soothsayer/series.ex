defmodule Soothsayer.Series do
  @moduledoc """
  Several series in one model.

  With `series: %{column: "id"}` the training frame holds many series told
  apart by that column, and one network is trained on all of them. What the
  series share and what they keep for themselves:

    * Shared: the frequency (every series must have the same), the time
      axis and changepoints (from the union of all timestamps, NeuralProphet's
      `global_time_normalization`), the `:auto` seasonality decisions,
      holidays and events, and the network weights.
    * Per series: the `y` scale under `normalize: :local` (the default,
      like NeuralProphet), the observed values that seed the lags, and the
      regressor and condition values.

  Every training sample carries two extra inputs: `"series"`, a one-hot of
  the series it comes from, and `"series_level"`, the series' mean over its
  standard deviation, which is the level multiplicative components scale
  with in normalized space (a constant for a single series, see
  `Soothsayer.Model`).

  Prediction takes a frame with `ds` and the id column, and so do the
  `:history` and `:regressors` frames. Ids not seen at fit raise.
  """

  alias Explorer.DataFrame

  @doc "The id column, `nil` for a single series model."
  @spec column(map()) :: String.t() | nil
  def column(config), do: get_in(config, [:series, :column])

  @doc "The sorted ids the model was fitted on, `nil` before fit or for a single series."
  @spec ids(map()) :: list(String.t()) | nil
  def ids(config), do: get_in(config, [:series, :ids])

  @doc "Whether the config asks for several series."
  @spec enabled?(map()) :: boolean()
  def enabled?(config), do: column(config) != nil

  @doc """
  Validates the `series` config: `column` nil or a string, `normalize`
  `:local` or `:global`, `trend` and `seasonality` `:global` or `:local`
  (only with a column), `local_regularization` nil or a non-negative
  number (only with something local).
  """
  @spec validate_config!(map()) :: :ok
  def validate_config!(%{series: series}) do
    unless well_formed?(series) do
      raise ArgumentError,
            "series must be %{column: nil | \"name\", normalize: :local | :global, " <>
              "trend: :global | :local, seasonality: :global | :local, " <>
              "local_regularization: nil | number >= 0}, got #{inspect(series)}"
    end

    if is_nil(series.column) and :local in [series.trend, series.seasonality] do
      raise ArgumentError, "series.trend and series.seasonality can only be :local with a column"
    end

    if series.local_regularization not in [nil, 0] and
         :local not in [series.trend, series.seasonality] do
      raise ArgumentError,
            "series.local_regularization needs a :local trend or seasonality to pull together"
    end

    :ok
  end

  defp well_formed?(%{
         column: column,
         normalize: normalize,
         trend: trend,
         seasonality: seasonality,
         local_regularization: local_regularization
       }) do
    (is_nil(column) or is_binary(column)) and normalize in [:local, :global] and
      trend in [:global, :local] and seasonality in [:global, :local] and
      (is_nil(local_regularization) or
         (is_number(local_regularization) and local_regularization >= 0))
  end

  defp well_formed?(_series), do: false

  @doc """
  The ids when `component` (`:trend` or `:seasonality`) is `:local` and the
  ids are known, else `nil`: what a component checks to decide between one
  shared kernel and one per series.
  """
  @spec local_ids(map(), :trend | :seasonality) :: list(String.t()) | nil
  def local_ids(config, component) do
    if get_in(config, [:series, component]) == :local, do: ids(config)
  end

  @doc """
  The layer names with one kernel per series, for the local regularization.
  """
  @spec local_layers(map()) :: list(String.t())
  def local_layers(config) do
    trend = if local_ids(config, :trend), do: ["trend_dense"], else: []

    seasonality =
      if local_ids(config, :seasonality),
        do: Enum.map(Soothsayer.Seasonality.periods(config), &"#{&1}_dense"),
        else: []

    trend ++ seasonality
  end

  @doc """
  Splits tensors with a leading series axis into a map by id: `%{kernel:
  {n, ...}}` becomes `%{"a" => %{kernel: ...}, ...}`.
  """
  @spec by_id(list(String.t()), %{atom() => Nx.Tensor.t() | nil}) :: %{String.t() => map()}
  def by_id(ids, tensors) do
    ids
    |> Enum.with_index()
    |> Map.new(fn {id, index} ->
      {id, Map.new(tensors, fn {key, tensor} -> {key, tensor && tensor[index]} end)}
    end)
  end

  @doc """
  Splits a frame into `{id, frame}` pairs, one per series, ids sorted. A
  single series model gets the whole frame under `nil`. Ids must be strings
  and every series needs at least two rows.
  """
  @spec split(DataFrame.t(), map()) :: list({String.t() | nil, DataFrame.t()})
  def split(%DataFrame{} = frame, config) do
    case column(config) do
      nil ->
        [{nil, frame}]

      column ->
        Enum.map(unique_ids!(frame, column), &{&1, series_rows!(frame, column, &1)})
    end
  end

  defp series_rows!(frame, column, id) do
    rows = rows_of(frame, column, id)

    if DataFrame.n_rows(rows) < 2 do
      raise ArgumentError, "Series #{inspect(id)} needs at least 2 rows"
    end

    rows
  end

  @doc """
  The rows of a frame belonging to one series, in their original order.
  """
  @spec rows_of(DataFrame.t(), String.t(), String.t()) :: DataFrame.t()
  def rows_of(%DataFrame{} = frame, column, id) do
    DataFrame.filter_with(frame, &Explorer.Series.equal(&1[column], id))
  end

  @doc """
  The distinct ids of a frame's id column, sorted. Raises when the column is
  missing or holds anything but strings.
  """
  @spec unique_ids!(DataFrame.t(), String.t()) :: list(String.t())
  def unique_ids!(%DataFrame{} = frame, column) do
    unless column in DataFrame.names(frame) do
      raise ArgumentError,
            "Series column #{inspect(column)} not found. " <>
              "Available columns: #{inspect(DataFrame.names(frame))}"
    end

    ids = frame[column] |> Explorer.Series.distinct() |> Explorer.Series.to_list()

    for id <- ids, not is_binary(id) do
      raise ArgumentError, "Series ids must be strings, got #{inspect(id)} in #{inspect(column)}"
    end

    Enum.sort(ids)
  end

  @doc """
  The two series inputs for `rows` samples of series `id`: the one-hot
  `{rows, n}` and the level `{rows, 1}`. Returns `x` unchanged for a single
  series model.
  """
  @spec put_inputs(map(), map(), String.t() | nil, map(), pos_integer()) :: map()
  def put_inputs(x, config, id, entry, rows) do
    case ids(config) do
      nil ->
        x

      ids ->
        index = Enum.find_index(ids, &(&1 == id))
        one_hot = Nx.equal(Nx.iota({1, length(ids)}), index) |> Nx.as_type({:f, 32})
        %{mean: mean, std: std} = entry.normalization
        level = Nx.divide(mean, std) |> Nx.reshape({1, 1})

        x
        |> Map.put("series", Nx.broadcast(one_hot, {rows, length(ids)}))
        |> Map.put("series_level", Nx.broadcast(level, {rows, 1}))
    end
  end

  @doc """
  The network inputs, `nil` until the ids are known (fit rebuilds the
  network once they are).
  """
  @spec build_network_inputs(map()) :: %{series: Axon.t(), level: Axon.t()} | nil
  def build_network_inputs(config) do
    case ids(config) do
      nil ->
        nil

      ids ->
        %{
          series: Axon.input("series", shape: {nil, length(ids)}),
          level: Axon.input("series_level", shape: {nil, 1})
        }
    end
  end
end
