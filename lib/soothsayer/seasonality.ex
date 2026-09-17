defmodule Soothsayer.Seasonality do
  @moduledoc """
  Seasonality component for Soothsayer forecasting models.

  Handles network building and feature engineering for yearly, weekly and
  daily seasonality using Fourier series decomposition. Each period gets
  `2 * fourier_terms` features, a sine and a cosine per term, computed from
  where the timestamp falls within the year, the week or the day.

  Each period's `enabled` can be `true`, `false` or `:auto`. `:auto` is
  resolved at fit with NeuralProphet's rules, see `resolve_auto/3`. Until
  then it counts as disabled.

  Custom seasonalities live under `seasonality.custom`, keyed by name with a
  `period` in days and `fourier_terms`, like NeuralProphet's
  `add_seasonality`. They are always enabled and their phase is measured
  from 1900-01-01, as upstream does. Any period, built in or custom, can
  carry a `condition`: the name of a 0 to 1 column in the data (booleans
  work) that scales its features row by row, so a weekly pattern can exist
  only in summer. Condition columns travel with the regressors: in the
  training dataframe at fit and in the `regressors:` dataframe at predict.
  """

  alias Explorer.DataFrame
  alias Soothsayer.AR
  alias Soothsayer.Frequency
  alias Soothsayer.Layers
  alias Soothsayer.Regressors
  alias Soothsayer.Timestamp

  @periods [:yearly, :weekly, :daily]
  @custom_epoch ~D[1900-01-01]
  @custom_name ~r/^[a-z][a-z0-9_]*$/
  @config_keys [:mode, :regularization, :custom | @periods]

  # Calendar constants
  @days_per_regular_year 365.0
  @days_per_leap_year 366.0
  @days_per_week 7.0

  @doc """
  The built-in seasonal periods, in the order the network and the
  components use.
  """
  @spec periods() :: list(atom())
  def periods, do: @periods

  @doc """
  Every period of a config, the built-in ones followed by the custom ones
  as atoms, sorted by name.

  ## Examples

      iex> Soothsayer.Seasonality.periods(%{seasonality: %{custom: %{"monthly" => %{period: 30.5, fourier_terms: 3}}}})
      [:yearly, :weekly, :daily, :monthly]

  """
  @spec periods(map()) :: list(atom())
  def periods(config) do
    @periods ++ Enum.map(custom_names(config), &String.to_atom/1)
  end

  defp custom_names(config) do
    (get_in(config, [:seasonality, :custom]) || %{}) |> Map.keys() |> Enum.sort()
  end

  @doc """
  Validates the `seasonality` config: known keys only, custom seasonalities
  with a positive `period` in days, a positive integer `fourier_terms` and
  a name matching `#{inspect(@custom_name)}` that is not a built-in period,
  and `condition` either absent or a column name.
  """
  @spec validate_config!(map()) :: :ok
  def validate_config!(seasonality) when is_map(seasonality) do
    for key <- Map.keys(seasonality), key not in @config_keys do
      raise ArgumentError,
            "unknown seasonality key #{inspect(key)}. Known: #{inspect(@config_keys)}. " <>
              "Other periods go under seasonality.custom, like " <>
              "custom: %{\"monthly\" => %{period: 30.5, fourier_terms: 3}}"
    end

    for period <- @periods, spec = seasonality[period], do: validate_condition!(period, spec)

    for {name, spec} <- Map.get(seasonality, :custom, %{}) do
      validate_custom!(name, spec)
    end

    :ok
  end

  def validate_config!(seasonality) do
    raise ArgumentError, "seasonality must be a map, got #{inspect(seasonality)}"
  end

  defp validate_custom!(name, spec) do
    unless is_binary(name) and Regex.match?(@custom_name, name) and
             name not in Enum.map(@periods, &Atom.to_string/1) do
      raise ArgumentError,
            "custom seasonality names are lowercase identifiers other than yearly, weekly " <>
              "and daily, got #{inspect(name)}"
    end

    unless is_map(spec) and is_number(spec[:period]) and spec[:period] > 0 do
      raise ArgumentError,
            "seasonality.custom.#{name}.period must be a positive number of days, " <>
              "got #{inspect(spec[:period])}"
    end

    unless is_integer(spec[:fourier_terms]) and spec[:fourier_terms] > 0 do
      raise ArgumentError,
            "seasonality.custom.#{name}.fourier_terms must be a positive integer, " <>
              "got #{inspect(spec[:fourier_terms])}"
    end

    validate_condition!(name, spec)
  end

  defp validate_condition!(period, spec) do
    condition = Map.get(spec, :condition)

    unless is_nil(condition) or is_binary(condition) do
      raise ArgumentError,
            "seasonality.#{period}.condition must be a column name, got #{inspect(condition)}"
    end
  end

  @doc """
  The spec of a period: its own map for a built-in period, and for a custom
  period its map with `enabled: true`. `nil` when the config has no such
  period.
  """
  @spec period_spec(map(), atom()) :: map() | nil
  def period_spec(config, period) when period in @periods do
    get_in(config, [:seasonality, period])
  end

  def period_spec(config, period) do
    case get_in(config, [:seasonality, :custom, Atom.to_string(period)]) do
      nil -> nil
      spec -> Map.put(spec, :enabled, true)
    end
  end

  @doc """
  The condition column names a config uses, sorted and unique.
  """
  @spec condition_columns(map()) :: list(String.t())
  def condition_columns(config) do
    config
    |> periods()
    |> Enum.map(&get_in(period_spec(config, &1) || %{}, [:condition]))
    |> Enum.reject(&is_nil/1)
    |> Enum.uniq()
    |> Enum.sort()
  end

  # Network Building

  @doc """
  Creates Axon input nodes for seasonality components.

  ## Parameters

    * `config` - Model configuration map with `:seasonality` key.

  ## Returns

    A map with an Axon input node per configured period (`:yearly`,
    `:weekly`, `:daily`), each `{nil, positions, 2 * fourier_terms}` where
    `positions` is the number of timestamps in a sample, see
    `Soothsayer.AR.positions/1`. Periods missing from the config get no input.

  """
  @spec build_inputs(map()) :: %{optional(atom()) => Axon.t()}
  def build_inputs(config) do
    positions = AR.positions(config)

    Map.new(configured_periods(config), fn period ->
      terms = fourier_terms(config, period)
      {period, Axon.input(Atom.to_string(period), shape: {nil, positions, terms * 2})}
    end)
  end

  defp configured_periods(config) do
    Enum.filter(periods(config), &(period_spec(config, &1) != nil))
  end

  defp fourier_terms(config, period) do
    get_in(period_spec(config, period) || %{}, [:fourier_terms]) || 0
  end

  @doc """
  Builds seasonality component layers.

  ## Parameters

    * `inputs` - Map of Axon input nodes from `build_inputs/1`.
    * `config` - Model configuration map.

  ## Returns

    A map with an Axon layer per input, keyed like `inputs`, each a linear
    layer over every position (`{batch, positions}`) or `Axon.constant(0)`
    for a disabled period.

  """
  @spec build_components(%{optional(atom()) => Axon.t()}, map()) ::
          %{optional(atom()) => Axon.t()}
  def build_components(inputs, config, series_input \\ nil) do
    Map.new(inputs, fn {period, input} ->
      {period, build_period_component(input, config, period, series_input)}
    end)
  end

  defp build_period_component(input, config, period, series_input) do
    local_ids = Soothsayer.Series.local_ids(config, :seasonality)

    cond do
      not enabled?(config, period) ->
        Axon.constant(0)

      local_ids ->
        Layers.series_dense(input, series_input, length(local_ids), "#{period}_dense")

      true ->
        Layers.position_dense(input, "#{period}_dense")
    end
  end

  @doc """
  Whether a period is enabled in the config. `:auto` counts as disabled
  until `resolve_auto/3` has run.
  """
  @spec enabled?(map(), atom()) :: boolean()
  def enabled?(config, period), do: get_in(period_spec(config, period) || %{}, [:enabled]) == true

  @doc """
  Resolves every `enabled: :auto` in a seasonality config to `true` or
  `false` from the data, with the rules NeuralProphet uses: yearly needs at
  least two years of data, weekly at least two weeks and a step shorter than
  a week, daily at least two days and a step shorter than a day.

  ## Examples

      iex> config = %{yearly: %{enabled: :auto, fourier_terms: 6}, weekly: %{enabled: true, fourier_terms: 3}, daily: %{enabled: :auto, fourier_terms: 6}}
      iex> timestamps = [~N[2023-01-01 00:00:00], ~N[2023-01-04 00:00:00]]
      iex> resolved = Soothsayer.Seasonality.resolve_auto(config, timestamps, {1, :hour})
      iex> {resolved.yearly.enabled, resolved.weekly.enabled, resolved.daily.enabled}
      {false, true, true}

  """
  @spec resolve_auto(map(), list(Timestamp.input()), Frequency.t()) :: map()
  def resolve_auto(seasonality_config, timestamps, frequency) do
    span_days = Timestamp.days_since(List.last(timestamps), List.first(timestamps))
    step_days = Frequency.to_days(frequency)

    auto_enabled = %{
      yearly: span_days >= 730,
      weekly: span_days >= 14 and step_days < 7,
      daily: span_days >= 2 and step_days < 1
    }

    Enum.reduce(@periods, seasonality_config, fn period, config ->
      case config[period] do
        %{enabled: :auto} -> put_in(config, [period, :enabled], auto_enabled[period])
        _ -> config
      end
    end)
  end

  # Feature Engineering

  @doc """
  Builds seasonality feature tensors from timestamps.

  ## Parameters

    * `timestamps` - List of dates or naive datetimes.
    * `config` - Model configuration map with `:seasonality` key.
    * `conditions` - Condition values by column name and naive timestamp,
      see `condition_values/2`. Needed for every period with a `condition`.
    * `opts` - `:required`, a `MapSet` of the naive timestamps that must
      have a condition value; the others get 0 when missing. Defaults to
      all of them.

  ## Returns

    A map with a tensor of shape `{n, 2 * fourier_terms}` per configured
    period, all zeros for disabled periods, and zero rows where a
    condition is 0.

  ## Examples

      iex> dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]
      iex> config = %{seasonality: %{yearly: %{enabled: true, fourier_terms: 2}, weekly: %{enabled: true, fourier_terms: 2}}}
      iex> result = Soothsayer.Seasonality.build_features(dates, config)
      iex> Nx.shape(result.yearly)
      {3, 4}

  """
  @spec build_features(
          list(Timestamp.input()),
          map(),
          %{String.t() => %{NaiveDateTime.t() => float()}},
          keyword()
        ) :: %{optional(atom()) => Nx.Tensor.t()}
  def build_features(timestamps, config, conditions \\ %{}, opts \\ []) do
    naive = Enum.map(timestamps, &Timestamp.to_naive_datetime/1)
    required = Keyword.get(opts, :required)

    Map.new(configured_periods(config), fn period ->
      spec = period_spec(config, period)
      terms = fourier_terms(config, period)

      features =
        if enabled?(config, period) do
          naive
          |> build_period_features(period, spec, terms)
          |> apply_condition(naive, spec[:condition], conditions, required)
        else
          Nx.broadcast(0.0, {length(naive), terms * 2}) |> Nx.as_type({:f, 32})
        end

      {period, features}
    end)
  end

  defp apply_condition(features, _naive, nil, _conditions, _required), do: features

  defp apply_condition(features, naive, column, conditions, required) do
    values = Map.get(conditions, column, %{})

    mask = Enum.map(naive, &condition_value!(values, column, &1, required))
    Nx.multiply(features, Nx.tensor(mask, type: {:f, 32}) |> Nx.reshape({:auto, 1}))
  end

  defp condition_value!(values, column, timestamp, required) do
    case Map.fetch(values, timestamp) do
      {:ok, value} ->
        value

      :error when required == nil ->
        raise_missing_condition!(column, timestamp)

      :error ->
        if MapSet.member?(required, timestamp),
          do: raise_missing_condition!(column, timestamp),
          else: 0.0
    end
  end

  defp raise_missing_condition!(column, timestamp) do
    raise ArgumentError,
          "Seasonality condition #{inspect(column)} has no value for " <>
            "#{Timestamp.format(timestamp)}. The column must cover every timestamp " <>
            "being fitted or predicted."
  end

  @doc """
  Raises `ArgumentError` when a condition column of the config is missing
  from the dataframe.
  """
  @spec validate_condition_columns!(DataFrame.t(), map()) :: :ok
  def validate_condition_columns!(%DataFrame{} = dataframe, config) do
    columns = DataFrame.names(dataframe)

    for column <- condition_columns(config), column not in columns do
      raise ArgumentError,
            "Seasonality condition column #{inspect(column)} not found. " <>
              "Available columns: #{inspect(columns)}"
    end

    :ok
  end

  @doc """
  Condition values by column name and naive timestamp from a dataframe with
  a "ds" column, for every condition column of the config. Booleans become
  1.0 and 0.0. Raises `ArgumentError` when a column is missing or has a
  value outside 0 to 1.
  """
  @spec condition_values(DataFrame.t(), map()) ::
          %{String.t() => %{NaiveDateTime.t() => float()}}
  def condition_values(%DataFrame{} = dataframe, config) do
    validate_condition_columns!(dataframe, config)

    Map.new(condition_columns(config), fn column ->
      values = Regressors.values_by_timestamp(dataframe, column)

      for {_timestamp, value} <- values, value < 0 or value > 1 do
        raise ArgumentError,
              "Seasonality condition #{inspect(column)} must be between 0 and 1 or boolean, " <>
                "got #{inspect(value)}"
      end

      {column, values}
    end)
  end

  # Columns are sin_1, cos_1, sin_2, cos_2, ... The angles are computed in
  # f64: the yearly angle reaches 2 * pi * 6 at the highest term, where f32
  # has already lost more precision than the sine is worth.
  defp build_period_features(timestamps, period, spec, fourier_terms) do
    fractions =
      if period in @periods do
        compute_period_fractions(timestamps, period)
      else
        compute_custom_fractions(timestamps, spec.period)
      end

    fractions = Nx.tensor([fractions], type: {:f, 64})
    terms = Nx.tensor([Enum.to_list(1..fourier_terms)], type: {:f, 64})

    angles =
      fractions
      |> Nx.transpose()
      |> Nx.multiply(terms)
      |> Nx.multiply(2 * :math.pi())

    Nx.stack([Nx.sin(angles), Nx.cos(angles)], axis: -1)
    |> Nx.reshape({length(timestamps), 2 * fourier_terms})
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Where each timestamp falls within its period, as a fraction. At midnight
  the yearly and weekly fractions are the same as for the plain date, so
  daily data is unaffected by the time of day term.

  ## Examples

      iex> Soothsayer.Seasonality.compute_period_fractions([~N[2023-01-01 12:00:00]], :daily)
      [0.5]

      iex> Soothsayer.Seasonality.compute_period_fractions([~D[2023-01-02]], :weekly)
      [1 / 7]

  """
  @spec compute_period_fractions(list(Timestamp.input()), atom()) :: list(float())
  def compute_period_fractions(timestamps, :yearly) do
    Enum.map(timestamps, fn timestamp ->
      naive = Timestamp.to_naive_datetime(timestamp)
      date = NaiveDateTime.to_date(naive)

      days_in_year =
        if Date.leap_year?(date), do: @days_per_leap_year, else: @days_per_regular_year

      (Date.day_of_year(date) + Timestamp.time_of_day(naive)) / days_in_year
    end)
  end

  def compute_period_fractions(timestamps, :weekly) do
    Enum.map(timestamps, fn timestamp ->
      naive = Timestamp.to_naive_datetime(timestamp)
      day_of_week = naive |> NaiveDateTime.to_date() |> Date.day_of_week()
      (day_of_week + Timestamp.time_of_day(naive)) / @days_per_week
    end)
  end

  def compute_period_fractions(timestamps, :daily) do
    Enum.map(timestamps, &Timestamp.time_of_day/1)
  end

  @doc """
  Where each timestamp falls within a custom period of `period_days`,
  counted from 1900-01-01 like NeuralProphet does.

  ## Examples

      iex> Soothsayer.Seasonality.compute_custom_fractions([~D[1900-01-01], ~N[1900-01-16 06:00:00]], 30.5)
      [0.0, 0.5]

  """
  @spec compute_custom_fractions(list(Timestamp.input()), number()) :: list(float())
  def compute_custom_fractions(timestamps, period_days) do
    Enum.map(timestamps, fn timestamp ->
      :math.fmod(Timestamp.days_since(timestamp, @custom_epoch), period_days) / period_days
    end)
  end
end
