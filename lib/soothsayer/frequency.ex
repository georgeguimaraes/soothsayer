defmodule Soothsayer.Frequency do
  @moduledoc """
  The spacing between consecutive rows of a series, `{amount, unit}` with
  units `:minute`, `:hour`, `:day` or `:month`: `{1, :day}` for daily data,
  `{5, :minute}` for five minute readings, `{1, :month}` for monthly totals.

  Every place that steps through time (auto-regression lags, forecast
  blocks, event windows, lagged regressor windows) moves by whole steps of
  the frequency, so an AR lag of 3 on hourly data means three hours back.

  The default `frequency: :auto` infers it at fit from the most common gap
  between consecutive timestamps, so a single missing row doesn't change the
  answer. Gaps of 28 to 31 days count as one month and 365 or 366 days as a
  year, since calendar months and years aren't a fixed number of seconds.
  """

  alias Soothsayer.Timestamp

  @type unit :: :minute | :hour | :day | :month
  @type t :: {pos_integer(), unit()}

  @units [:minute, :hour, :day, :month]
  @seconds_per_unit %{minute: 60, hour: 3_600, day: 86_400}
  @seconds_per_day 86_400
  @average_days_per_month 30.436875

  @doc """
  Raises `ArgumentError` unless `frequency` is `:auto` or a valid
  `{amount, unit}` tuple.
  """
  @spec validate!(term()) :: :ok
  def validate!(:auto), do: :ok

  def validate!({amount, unit}) when is_integer(amount) and amount > 0 and unit in @units do
    :ok
  end

  def validate!(other) do
    raise ArgumentError,
          "frequency must be :auto or {amount, unit} with a positive integer amount and " <>
            "a unit in #{inspect(@units)}, got #{inspect(other)}"
  end

  @doc """
  Infers the frequency from a sorted list of timestamps, using the most
  common gap between consecutive ones.

  ## Examples

      iex> Soothsayer.Frequency.infer([~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]])
      {1, :day}

      iex> Soothsayer.Frequency.infer([~N[2023-01-01 00:00:00], ~N[2023-01-01 00:05:00], ~N[2023-01-01 00:10:00]])
      {5, :minute}

      iex> Soothsayer.Frequency.infer([~D[2023-01-01], ~D[2023-02-01], ~D[2023-03-01]])
      {1, :month}

  """
  @spec infer(list(Timestamp.input())) :: t()
  def infer(timestamps) when length(timestamps) < 2 do
    raise ArgumentError, "Inferring a frequency needs at least 2 timestamps"
  end

  def infer(timestamps) do
    naive = Enum.map(timestamps, &Timestamp.to_naive_datetime/1)

    {seconds, _count} =
      naive
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [previous, current] -> NaiveDateTime.diff(current, previous, :second) end)
      |> Enum.frequencies()
      |> Enum.max_by(fn {seconds, count} -> {count, -seconds} end)

    from_seconds(seconds)
  end

  defp from_seconds(seconds) when seconds <= 0 do
    raise ArgumentError,
          "Timestamps must be strictly increasing to infer a frequency, " <>
            "found a step of #{seconds} seconds"
  end

  defp from_seconds(seconds) do
    days = seconds / @seconds_per_day

    cond do
      days >= 365 and days <= 366 ->
        {12, :month}

      days >= 28 and days <= 31 ->
        {1, :month}

      rem(seconds, @seconds_per_unit.day) == 0 ->
        {div(seconds, @seconds_per_unit.day), :day}

      rem(seconds, @seconds_per_unit.hour) == 0 ->
        {div(seconds, @seconds_per_unit.hour), :hour}

      rem(seconds, @seconds_per_unit.minute) == 0 ->
        {div(seconds, 60), :minute}

      true ->
        raise ArgumentError,
              "Cannot infer a frequency from a step of #{seconds} seconds. " <>
                "Pass frequency: {amount, unit} in the model config."
    end
  end

  @doc """
  Moves a timestamp by `steps` steps of the frequency, negative steps
  going back. A `Date` stays a `Date` for day and month frequencies and
  becomes a `NaiveDateTime` for shorter ones. Month steps keep the day of
  the month, and a timestamp on the last day of its month lands on the last
  day of the target month, so month-end series stay on their own grid.

  ## Examples

      iex> Soothsayer.Frequency.shift(~D[2023-01-31], 1, {1, :month})
      ~D[2023-02-28]

      iex> Soothsayer.Frequency.shift(~D[2023-02-28], 1, {1, :month})
      ~D[2023-03-31]

      iex> Soothsayer.Frequency.shift(~D[2023-02-28], -1, {1, :month})
      ~D[2023-01-31]

      iex> Soothsayer.Frequency.shift(~N[2023-01-01 00:00:00], -2, {5, :minute})
      ~N[2022-12-31 23:50:00]

      iex> Soothsayer.Frequency.shift(~D[2023-01-01], 3, {1, :day})
      ~D[2023-01-04]

  """
  @spec shift(Timestamp.input(), integer(), t()) :: Timestamp.input()
  def shift(%Date{} = date, steps, {amount, :day}), do: Date.add(date, steps * amount)

  def shift(%Date{} = date, steps, {amount, :month}) do
    shifted = Date.shift(date, month: steps * amount)
    if month_end?(date), do: Date.end_of_month(shifted), else: shifted
  end

  def shift(%Date{} = date, steps, frequency) do
    shift(Timestamp.to_naive_datetime(date), steps, frequency)
  end

  def shift(%NaiveDateTime{} = timestamp, steps, {amount, :month}) do
    shifted = NaiveDateTime.shift(timestamp, month: steps * amount)

    if month_end?(NaiveDateTime.to_date(timestamp)) do
      NaiveDateTime.new!(
        shifted |> NaiveDateTime.to_date() |> Date.end_of_month(),
        NaiveDateTime.to_time(shifted)
      )
    else
      shifted
    end
  end

  def shift(%NaiveDateTime{} = timestamp, steps, {amount, unit}) do
    NaiveDateTime.add(timestamp, steps * amount, unit)
  end

  @doc """
  How many steps of the frequency lead from `from` to `to`, negative when
  `to` comes first. Raises `ArgumentError` when `to` doesn't sit a whole
  number of steps away, since a lag or forecast block can't be built for a
  timestamp that is off the grid.

  ## Examples

      iex> Soothsayer.Frequency.steps_between(~D[2023-01-01], ~D[2023-01-08], {1, :day})
      7

      iex> Soothsayer.Frequency.steps_between(~N[2023-01-01 06:00:00], ~N[2023-01-01 00:00:00], {1, :hour})
      -6

      iex> Soothsayer.Frequency.steps_between(~D[2023-01-31], ~D[2023-03-31], {1, :month})
      2

  """
  @spec steps_between(Timestamp.input(), Timestamp.input(), t()) :: integer()
  def steps_between(from, to, {amount, :month} = frequency) do
    from_naive = Timestamp.to_naive_datetime(from)
    to_naive = Timestamp.to_naive_datetime(to)
    months = (to_naive.year - from_naive.year) * 12 + (to_naive.month - from_naive.month)

    with 0 <- rem(months, amount),
         steps = div(months, amount),
         true <- shift(from_naive, steps, frequency) == to_naive do
      steps
    else
      _ -> raise_off_grid(from, to, frequency)
    end
  end

  def steps_between(from, to, {amount, unit} = frequency) do
    seconds =
      NaiveDateTime.diff(
        Timestamp.to_naive_datetime(to),
        Timestamp.to_naive_datetime(from),
        :second
      )

    step_seconds = amount * @seconds_per_unit[unit]

    case rem(seconds, step_seconds) do
      0 -> div(seconds, step_seconds)
      _ -> raise_off_grid(from, to, frequency)
    end
  end

  defp month_end?(%Date{} = date), do: date == Date.end_of_month(date)

  defp raise_off_grid(from, to, frequency) do
    raise ArgumentError,
          "#{Timestamp.format(to)} is not a whole number of #{describe(frequency)} steps " <>
            "from #{Timestamp.format(from)}. Auto-regression needs timestamps on the " <>
            "same #{describe(frequency)} grid as the training data."
  end

  @doc """
  The timestamps after `from`, one step apart, up to and including `to`.
  Empty when `to` is not after `from`. Each one is shifted directly from
  `from`, so month ends don't drift.

  ## Examples

      iex> Soothsayer.Frequency.range(~D[2023-01-31], ~D[2023-04-30], {1, :month})
      [~D[2023-02-28], ~D[2023-03-31], ~D[2023-04-30]]

      iex> Soothsayer.Frequency.range(~D[2023-01-03], ~D[2023-01-01], {1, :day})
      []

  """
  @spec range(Timestamp.input(), Timestamp.input(), t()) :: list(Timestamp.input())
  def range(from, to, frequency) do
    case steps_between(from, to, frequency) do
      steps when steps > 0 -> Enum.map(1..steps, &shift(from, &1, frequency))
      _ -> []
    end
  end

  @doc """
  The length of one step in days, an average for months. Used to decide
  which seasonalities make sense for the data.

  ## Examples

      iex> Soothsayer.Frequency.to_days({6, :hour})
      0.25

  """
  @spec to_days(t()) :: float()
  def to_days({amount, :month}), do: amount * @average_days_per_month
  def to_days({amount, unit}), do: amount * @seconds_per_unit[unit] / @seconds_per_day

  @doc """
  A readable form for messages, like "5 minute" or "1 day".
  """
  @spec describe(t()) :: String.t()
  def describe({amount, unit}), do: "#{amount} #{unit}"
end
