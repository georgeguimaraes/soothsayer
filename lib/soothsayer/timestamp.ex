defmodule Soothsayer.Timestamp do
  @moduledoc """
  Timestamps in Soothsayer.

  Every `ds` value becomes a `NaiveDateTime` with second precision once it
  enters the library, whether it came from a `:date` or a
  `{:naive_datetime, _}` Explorer series. A `Date` means midnight of that
  day, so a dataframe of event dates still lines up with hourly data. The
  helpers here do that conversion and the small pieces of calendar math the
  components need: days since a reference point for the trend and the
  fraction of the day for seasonality.
  """

  alias Explorer.Series

  @type t :: NaiveDateTime.t()
  @type input :: Date.t() | NaiveDateTime.t()

  @seconds_per_day 86_400

  @doc """
  Converts a `Date` or `NaiveDateTime` to a second precision `NaiveDateTime`.

  ## Examples

      iex> Soothsayer.Timestamp.to_naive_datetime(~D[2023-01-05])
      ~N[2023-01-05 00:00:00]

      iex> Soothsayer.Timestamp.to_naive_datetime(~N[2023-01-05 10:30:00.000000])
      ~N[2023-01-05 10:30:00]

  """
  @spec to_naive_datetime(input()) :: t()
  def to_naive_datetime(%NaiveDateTime{} = timestamp) do
    NaiveDateTime.truncate(timestamp, :second)
  end

  def to_naive_datetime(%Date{} = date), do: NaiveDateTime.new!(date, ~T[00:00:00])

  def to_naive_datetime(other) do
    raise ArgumentError,
          "Expected a Date or NaiveDateTime, got #{inspect(other)}"
  end

  @doc """
  Reads a `:date` or `{:naive_datetime, _}` series as a list of timestamps.

  Raises `ArgumentError` for any other dtype, since a string or integer
  column can't be stepped through by a frequency.
  """
  @spec from_series(Series.t()) :: list(t())
  def from_series(%Series{} = series) do
    case Series.dtype(series) do
      :date ->
        series |> Series.to_list() |> Enum.map(&to_naive_datetime/1)

      {:naive_datetime, _precision} ->
        series |> Series.to_list() |> Enum.map(&to_naive_datetime/1)

      dtype ->
        raise ArgumentError,
              "The ds column must be a date or naive datetime series, got #{inspect(dtype)}. " <>
                "Cast it with Explorer.Series.cast/2 or parse the CSV with " <>
                "dtypes: [{\"ds\", {:naive_datetime, :microsecond}}]."
    end
  end

  @doc """
  Days from `reference` to `timestamp` as a float, negative when the
  timestamp comes first.

  ## Examples

      iex> Soothsayer.Timestamp.days_since(~D[2023-01-03], ~D[2023-01-01])
      2.0

      iex> Soothsayer.Timestamp.days_since(~N[2023-01-01 12:00:00], ~D[2023-01-01])
      0.5

  """
  @spec days_since(input(), input()) :: float()
  def days_since(timestamp, reference) do
    seconds =
      NaiveDateTime.diff(to_naive_datetime(timestamp), to_naive_datetime(reference), :second)

    seconds / @seconds_per_day
  end

  @doc """
  The fraction of the day that has passed at `timestamp`, from 0.0 at
  midnight up to (but not including) 1.0.

  ## Examples

      iex> Soothsayer.Timestamp.time_of_day(~N[2023-01-01 06:00:00])
      0.25

      iex> Soothsayer.Timestamp.time_of_day(~D[2023-01-01])
      0.0

  """
  @spec time_of_day(input()) :: float()
  def time_of_day(timestamp) do
    %NaiveDateTime{hour: hour, minute: minute, second: second} = to_naive_datetime(timestamp)
    (hour * 3600 + minute * 60 + second) / @seconds_per_day
  end

  @doc """
  Formats a timestamp for error messages: the date alone at midnight, the
  full ISO 8601 form otherwise.

  ## Examples

      iex> Soothsayer.Timestamp.format(~N[2023-01-05 00:00:00])
      "2023-01-05"

      iex> Soothsayer.Timestamp.format(~N[2023-01-05 10:30:00])
      "2023-01-05T10:30:00"

  """
  @spec format(input()) :: String.t()
  def format(%Date{} = date), do: Date.to_iso8601(date)

  def format(timestamp) do
    naive = to_naive_datetime(timestamp)

    if time_of_day(naive) == 0.0 do
      naive |> NaiveDateTime.to_date() |> Date.to_iso8601()
    else
      NaiveDateTime.to_iso8601(naive)
    end
  end

  @doc """
  Raises `ArgumentError` unless the timestamps are strictly increasing.
  Every component assumes ordered rows and the frequency is inferred from
  consecutive differences, so a shuffled or duplicated `ds` would go wrong
  quietly.
  """
  @spec validate_sorted!(list(input())) :: :ok
  def validate_sorted!(timestamps) do
    timestamps
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.each(fn [previous, current] ->
      if NaiveDateTime.compare(to_naive_datetime(current), to_naive_datetime(previous)) != :gt do
        raise ArgumentError,
              "The ds column must be strictly increasing, found #{format(current)} " <>
                "after #{format(previous)}. Sort the dataframe by ds and drop duplicates."
      end
    end)
  end
end
