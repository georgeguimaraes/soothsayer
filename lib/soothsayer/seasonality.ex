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
  """

  alias Soothsayer.Frequency
  alias Soothsayer.Timestamp

  @periods [:yearly, :weekly, :daily]

  # Calendar constants
  @days_per_regular_year 365.0
  @days_per_leap_year 366.0
  @days_per_week 7.0

  @doc """
  The seasonal periods, in the order the network and the components use.
  """
  @spec periods() :: list(atom())
  def periods, do: @periods

  # Network Building

  @doc """
  Creates Axon input nodes for seasonality components.

  ## Parameters

    * `config` - Model configuration map with `:seasonality` key.

  ## Returns

    A map with an Axon input node per configured period (`:yearly`,
    `:weekly`, `:daily`). Periods missing from the config get no input.

  """
  @spec build_inputs(map()) :: %{optional(atom()) => Axon.t()}
  def build_inputs(config) do
    Map.new(configured_periods(config), fn period ->
      terms = get_in(config, [:seasonality, period, :fourier_terms]) || 0
      {period, Axon.input(Atom.to_string(period), shape: {nil, terms * 2})}
    end)
  end

  defp configured_periods(config) do
    seasonality = config[:seasonality] || %{}
    Enum.filter(@periods, &Map.has_key?(seasonality, &1))
  end

  @doc """
  Builds seasonality component layers.

  ## Parameters

    * `inputs` - Map of Axon input nodes from `build_inputs/1`.
    * `config` - Model configuration map.

  ## Returns

    A map with an Axon layer per input, keyed like `inputs`.

  """
  @spec build_components(%{optional(atom()) => Axon.t()}, map()) ::
          %{optional(atom()) => Axon.t()}
  def build_components(inputs, config) do
    Map.new(inputs, fn {period, input} ->
      {period, build_period_component(input, config, period)}
    end)
  end

  defp build_period_component(input, config, period) do
    if enabled?(config, period) do
      Axon.dense(input, 1, activation: :linear, name: "#{period}_dense")
    else
      Axon.constant(0)
    end
  end

  @doc """
  Whether a period is enabled in the config. `:auto` counts as disabled
  until `resolve_auto/3` has run.
  """
  @spec enabled?(map(), atom()) :: boolean()
  def enabled?(config, period), do: get_in(config, [:seasonality, period, :enabled]) == true

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

  ## Returns

    A map with a tensor of shape `{n, 2 * fourier_terms}` per configured
    period, all zeros for disabled periods.

  ## Examples

      iex> dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]
      iex> config = %{seasonality: %{yearly: %{enabled: true, fourier_terms: 2}, weekly: %{enabled: true, fourier_terms: 2}}}
      iex> result = Soothsayer.Seasonality.build_features(dates, config)
      iex> Nx.shape(result.yearly)
      {3, 4}

  """
  @spec build_features(list(Timestamp.input()), map()) :: %{optional(atom()) => Nx.Tensor.t()}
  def build_features(timestamps, config) do
    naive = Enum.map(timestamps, &Timestamp.to_naive_datetime/1)

    Map.new(configured_periods(config), fn period ->
      terms = get_in(config, [:seasonality, period, :fourier_terms]) || 0

      features =
        if enabled?(config, period) do
          build_period_features(naive, period, terms)
        else
          Nx.broadcast(0.0, {length(naive), terms * 2}) |> Nx.as_type({:f, 32})
        end

      {period, features}
    end)
  end

  defp build_period_features(timestamps, period, fourier_terms) do
    t = compute_period_fractions(timestamps, period)

    features =
      Enum.flat_map(1..fourier_terms, fn i ->
        sin_vals = Enum.map(t, fn t_val -> :math.sin(2 * :math.pi() * i * t_val) end)
        cos_vals = Enum.map(t, fn t_val -> :math.cos(2 * :math.pi() * i * t_val) end)
        [sin_vals, cos_vals]
      end)

    features
    |> Enum.map(&Nx.tensor/1)
    |> Nx.stack(axis: 1)
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
end
