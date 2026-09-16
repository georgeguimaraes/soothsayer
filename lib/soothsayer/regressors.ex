defmodule Soothsayer.Regressors do
  @moduledoc """
  Future regressor component for Soothsayer models.

  A future regressor is an external variable whose value is known for every
  date you want to forecast: temperature from a weather forecast, planned
  marketing spend, a price schedule. Each regressor becomes one input column
  feeding a linear layer, so its learned coefficient reads directly as the
  effect of one normalized unit of the regressor on the forecast.

  Regressors are configured as a list of column names, or as a map from
  column name to options, see `normalize_config!/1`. A regressor is
  `mode: :additive` (its coefficient is added to the forecast) or
  `mode: :multiplicative` (its coefficient is a fraction of the trend). The
  input columns are laid out additive regressors first, then multiplicative,
  each group sorted by name, and each mode gets its own linear layer,
  `regressors_dense` and `regressors_multiplicative_dense`. The training dataframe
  must contain those columns, and so must the dataframe passed as
  `regressors:` to `Soothsayer.predict/3`. The training values are kept on
  the fitted model, so with auto-regression the lag timestamps of a
  forecast can reach back into the training data without the prediction
  dataframe repeating them.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Layers
  alias Soothsayer.Timestamp

  @layer_name "regressors_dense"
  @multiplicative_layer_name "regressors_multiplicative_dense"
  @defaults %{mode: :additive, regularization: nil, layers: []}
  @known_keys Map.keys(@defaults)
  @modes [:additive, :multiplicative]

  # Configuration

  @doc """
  Normalizes the `regressors` config to a map from column name to a full
  spec. A list of names gives every regressor the defaults; a map may give
  each regressor part of its spec.

  ## Examples

      iex> Soothsayer.Regressors.normalize_config!(["temperature"])
      %{"temperature" => %{mode: :additive, regularization: nil, layers: []}}

      iex> Soothsayer.Regressors.normalize_config!(%{"temperature" => %{mode: :multiplicative}})
      %{"temperature" => %{mode: :multiplicative, regularization: nil, layers: []}}

  """
  @spec normalize_config!(list(String.t()) | %{String.t() => map()}) :: %{String.t() => map()}
  def normalize_config!(names) when is_list(names) do
    for name <- names, not is_binary(name) do
      raise ArgumentError, "regressors must be column name strings, got #{inspect(name)}"
    end

    Map.new(names, &{&1, @defaults})
  end

  def normalize_config!(specs) when is_map(specs) do
    Map.new(specs, fn {name, spec} ->
      unless is_binary(name) and is_map(spec) do
        raise ArgumentError,
              "regressors must map column name strings to option maps, " <>
                "got #{inspect(name)} => #{inspect(spec)}"
      end

      for key <- Map.keys(spec), key not in @known_keys do
        raise ArgumentError,
              "unknown option #{inspect(key)} for regressor #{inspect(name)}. " <>
                "Known: #{inspect(@known_keys)}"
      end

      spec = Map.merge(@defaults, spec)

      unless spec.mode in @modes do
        raise ArgumentError,
              "regressor #{inspect(name)} mode must be one of #{inspect(@modes)}, " <>
                "got #{inspect(spec.mode)}"
      end

      {name, spec}
    end)
  end

  def normalize_config!(other) do
    raise ArgumentError,
          "regressors must be a list of column names or a map of name to options, " <>
            "got #{inspect(other)}"
  end

  @doc """
  The configured regressor names in the column order of the regressors
  input: additive regressors first, then multiplicative, each sorted.
  """
  @spec names(map()) :: list(String.t())
  def names(%{regressors: regressors}) when is_map(regressors) do
    names_in(regressors, :additive) ++ names_in(regressors, :multiplicative)
  end

  def names(%{regressors: regressors}) when is_list(regressors), do: Enum.sort(regressors)
  def names(_config), do: []

  defp names_in(regressors, mode) do
    for {name, spec} <- Enum.sort(regressors), Map.get(spec, :mode, :additive) == mode, do: name
  end

  @doc """
  The column ranges of the regressors input by mode, `nil` for a mode with
  no columns.

  ## Examples

      iex> config = %{regressors: %{"a" => %{mode: :additive}, "b" => %{mode: :multiplicative}, "c" => %{mode: :additive}}}
      iex> Soothsayer.Regressors.mode_ranges(config)
      %{additive: 0..1, multiplicative: 2..2}

  """
  @spec mode_ranges(map()) :: %{additive: Range.t() | nil, multiplicative: Range.t() | nil}
  def mode_ranges(%{regressors: regressors} = config) when is_map(regressors) do
    additive_count = length(names_in(regressors, :additive))
    total = length(names(config))

    %{
      additive: range_from(0, additive_count),
      multiplicative: range_from(additive_count, total - additive_count)
    }
  end

  def mode_ranges(config),
    do: %{additive: range_from(0, length(names(config))), multiplicative: nil}

  defp range_from(_start, 0), do: nil
  defp range_from(start, count), do: start..(start + count - 1)

  # Network Building

  @doc """
  Creates the Axon input node for the regressors component.

  ## Returns

    An Axon input node `{nil, positions, regressors}` when regressors are
    configured (`positions` being the timestamps in a sample, see
    `Soothsayer.AR.positions/1`), `nil` when none are.

  """
  @spec build_network_input(map()) :: Axon.t() | nil
  def build_network_input(config) do
    case names(config) do
      [] -> nil
      names -> Axon.input("regressors", shape: {nil, AR.positions(config), length(names)})
    end
  end

  @doc """
  Builds the regressors component layer.

  ## Returns

    A linear layer over every position (`{batch, positions}`) when
    regressors are configured, `Axon.constant(0)` otherwise.

  """
  @spec build_component(Axon.t() | nil, map()) :: %{
          additive: Axon.t() | nil,
          multiplicative: Axon.t() | nil
        }
  def build_component(nil, _config), do: %{additive: nil, multiplicative: nil}

  def build_component(input, config) do
    ranges = mode_ranges(config)

    %{
      additive: Layers.position_dense_over(input, ranges.additive, @layer_name),
      multiplicative:
        Layers.position_dense_over(input, ranges.multiplicative, @multiplicative_layer_name)
    }
  end

  # Feature Engineering

  @doc """
  Builds the regressors input tensor, `{n, regressors}`, for a list of
  timestamps from a dataframe with a "ds" column plus one column per
  regressor, stacking the values in `names` order. Plain dates mean
  midnight, so they match a naive datetime "ds" column at that time.

  Raises `ArgumentError` when a regressor column is missing or when any
  timestamp has no row, since a forecast that silently fills in zeros for
  an unknown regressor value would be wrong without saying so. See
  `build_features/4` for the lookup-based variant used at prediction.

  ## Examples

      iex> dataframe = Explorer.DataFrame.new(%{"ds" => [~D[2023-01-01], ~D[2023-01-02]], "temperature" => [20.0, 22.5]})
      iex> Soothsayer.Regressors.build_features([~D[2023-01-02]], dataframe, ["temperature"]) |> Nx.to_flat_list()
      [22.5]

  """
  @spec build_features(list(Timestamp.input()), DataFrame.t(), list(String.t())) ::
          Nx.Tensor.t()
  def build_features(timestamps, %DataFrame{} = dataframe, names) do
    validate_columns!(dataframe, names)
    known_values = Map.new(names, &{&1, values_by_timestamp(dataframe, &1)})
    build_features(timestamps, known_values, names, [])
  end

  @doc """
  Builds the regressors input tensor, `{n, regressors}`, for a list of
  timestamps from known values by regressor name and timestamp, see
  `known_values/3`.

  ## Options

    * `:required` - a `MapSet` of the timestamps that must have a value.
      Defaults to all of them.
    * `:fill` - the values to use for timestamps outside `:required` that
      have none, one per regressor in `names` order. Without it every
      missing value raises.

  A sample's target positions run `forecast_steps` past its origin whether
  or not those timestamps were asked for, and a regressor value at a target
  position only feeds that position's own output, so filling the ones
  nobody asked for is harmless. Lag positions and requested timestamps
  must be present.
  """
  @spec build_features(
          list(Timestamp.input()),
          %{String.t() => %{Timestamp.t() => float()}},
          list(String.t()),
          keyword()
        ) :: Nx.Tensor.t()
  def build_features(timestamps, known_values, names, opts) when is_map(known_values) do
    required = Keyword.get(opts, :required)
    fill = Keyword.get(opts, :fill)
    fill_by_name = if fill, do: Enum.zip(names, fill) |> Map.new(), else: %{}

    rows =
      Enum.map(timestamps, fn timestamp ->
        key = Timestamp.to_naive_datetime(timestamp)

        fill_here =
          if required == nil or MapSet.member?(required, key), do: %{}, else: fill_by_name

        Enum.map(names, &fetch_value!(known_values[&1], &1, key, fill_here))
      end)

    rows |> Nx.tensor() |> Nx.as_type({:f, 32})
  end

  defp fetch_value!(values, name, timestamp, fill) do
    case {Map.fetch(values, timestamp), Map.fetch(fill, name)} do
      {{:ok, value}, _} ->
        value

      {:error, {:ok, fill_value}} ->
        fill_value

      {:error, :error} ->
        raise ArgumentError,
              "Regressor #{inspect(name)} has no value for #{Timestamp.format(timestamp)}. " <>
                "The regressors dataframe must cover every timestamp being predicted, " <>
                "including the steps between the last observation and the forecast."
    end
  end

  @doc """
  Collects the regressor values by name and timestamp for prediction: the
  training values stored on the model, overridden and extended by the
  `regressors:` dataframe when given.
  """
  @spec known_values(map(), DataFrame.t() | nil, list(String.t())) ::
          %{String.t() => %{Timestamp.t() => float()}}
  def known_values(training_data, regressors_df, names) do
    training_values = Map.get(training_data, :regressors, %{})

    Map.new(names, fn name ->
      from_frame =
        if regressors_df != nil and name in DataFrame.names(regressors_df) do
          values_by_timestamp(regressors_df, name)
        else
          %{}
        end

      {name, Map.merge(Map.get(training_values, name, %{}), from_frame)}
    end)
  end

  @doc """
  Raw values of one column by timestamp, for storing at fit time. Missing
  values (nil or NaN) are left out, so a lookup at those timestamps raises
  instead of feeding NaN to the network.
  """
  @spec values_by_timestamp(DataFrame.t(), String.t()) :: %{Timestamp.t() => float()}
  def values_by_timestamp(%DataFrame{} = dataframe, name) do
    timestamps = Timestamp.from_series(dataframe["ds"])
    values = dataframe[name] |> Series.cast({:f, 64}) |> Series.to_list()

    for {timestamp, value} <- Enum.zip(timestamps, values),
        not is_nil(value) and value != :nan,
        into: %{},
        do: {timestamp, value}
  end

  @doc """
  Raises `ArgumentError` unless `dataframe` has every regressor column.
  """
  @spec validate_columns!(DataFrame.t(), list(String.t())) :: :ok
  def validate_columns!(%DataFrame{} = dataframe, names) do
    columns = DataFrame.names(dataframe)

    for name <- names, name not in columns do
      raise ArgumentError,
            "Regressor column #{inspect(name)} not found. Available columns: #{inspect(columns)}"
    end

    :ok
  end

  # Weight Extraction

  @doc """
  Extracts the learned regressor coefficients from a fitted model.

  Coefficients of additive regressors are in normalized units: the change
  in normalized y for a one standard deviation change in the regressor.
  Coefficients of multiplicative regressors are fractions of the trend per
  standard deviation of the regressor.

  ## Returns

    A map from regressor name to coefficient.

  """
  @spec get_effects(Soothsayer.Model.t()) :: %{String.t() => float()}
  def get_effects(%Soothsayer.Model{} = model) do
    names = names(model.config)

    if names == [] do
      raise ArgumentError, "No regressors configured on this model"
    end

    unless model.params do
      raise ArgumentError, "Model has not been fitted yet"
    end

    coefficients =
      Enum.flat_map([@layer_name, @multiplicative_layer_name], fn layer ->
        case model.params.data[layer] do
          nil -> []
          %{"kernel" => kernel} -> Nx.to_flat_list(kernel)
        end
      end)

    Enum.zip(names, coefficients) |> Map.new()
  end
end
