defmodule Soothsayer.Events do
  @moduledoc """
  Event feature engineering for Soothsayer models.

  Events model special occasions (holidays, promotions, etc.) that affect the time series.
  Each event becomes binary features indicating whether the event occurs on a given date.
  """

  alias Explorer.DataFrame
  alias Explorer.Series

  @doc """
  Computes total number of event features based on config.

  Each event with window [lower_window, upper_window] creates
  |lower_window| + 1 + upper_window features.

  ## Examples

      iex> Events.n_features(%{})
      0

      iex> Events.n_features(%{"sale" => %{lower_window: 0, upper_window: 0}})
      1

      iex> Events.n_features(%{"black_friday" => %{lower_window: -2, upper_window: 1}})
      4

  """
  @spec n_features(map()) :: non_neg_integer()
  def n_features(events_config) when events_config == %{}, do: 0

  def n_features(events_config) do
    events_config
    |> Enum.map(fn {_name, %{lower_window: lower, upper_window: upper}} ->
      abs(lower) + 1 + upper
    end)
    |> Enum.sum()
  end

  @doc """
  Returns list of feature names for all configured events.

  Names are formatted as "event_name_offset" where offset indicates
  the window position relative to the event date.

  ## Examples

      iex> Events.feature_names(%{})
      []

      iex> Events.feature_names(%{"sale" => %{lower_window: 0, upper_window: 0}})
      ["sale_0"]

      iex> Events.feature_names(%{"bf" => %{lower_window: -1, upper_window: 1}})
      ["bf_-1", "bf_0", "bf_+1"]

  """
  @spec feature_names(map()) :: [String.t()]
  def feature_names(events_config) when events_config == %{}, do: []

  def feature_names(events_config) do
    events_config
    |> Enum.sort_by(fn {name, _} -> name end)
    |> Enum.flat_map(fn {name, %{lower_window: lower, upper_window: upper}} ->
      lower..upper
      |> Enum.map(fn offset ->
        offset_str = if offset > 0, do: "+#{offset}", else: "#{offset}"
        "#{name}_#{offset_str}"
      end)
    end)
  end

  @doc """
  Builds event features tensor from dates and events DataFrame.

  For each date, creates binary features indicating whether each event
  (with its window offsets) occurs on that date.

  ## Parameters

    * `dates` - An Explorer Series of dates.
    * `events_df` - A DataFrame with "event" and "ds" columns.
    * `events_config` - Map of event configurations.

  ## Returns

    A tensor of shape {n_dates, n_event_features} with 1.0 where
    events occur and 0.0 otherwise.

  ## Examples

      iex> dates = Explorer.Series.from_list([~D[2023-01-01], ~D[2023-01-02]])
      iex> events_df = Explorer.DataFrame.new(%{"event" => ["sale"], "ds" => [~D[2023-01-02]]})
      iex> config = %{"sale" => %{lower_window: 0, upper_window: 0}}
      iex> Events.build_features(dates, events_df, config)
      #Nx.Tensor<
        f32[2][1]
        [
          [0.0],
          [1.0]
        ]
      >

  """
  @spec build_features(Series.t(), DataFrame.t(), map()) :: Nx.Tensor.t() | nil
  def build_features(_dates, _events_df, events_config) when events_config == %{} do
    nil
  end

  def build_features(dates, events_df, events_config) do
    dates_list = Series.to_list(dates)

    # Build a map of event_name -> list of dates for quick lookup
    event_dates_map = build_event_dates_map(events_df)

    # For each event (sorted by name) and each window position, create a column
    columns =
      events_config
      |> Enum.sort_by(fn {name, _} -> name end)
      |> Enum.flat_map(fn {event_name, %{lower_window: lower, upper_window: upper}} ->
        event_dates = Map.get(event_dates_map, event_name, [])

        lower..upper
        |> Enum.map(fn offset ->
          build_feature_column(dates_list, event_dates, offset)
        end)
      end)

    if Enum.empty?(columns) do
      nil
    else
      columns
      |> Nx.stack(axis: 1)
    end
  end

  defp build_event_dates_map(events_df) do
    if DataFrame.n_rows(events_df) == 0 do
      %{}
    else
      event_names = events_df["event"] |> Series.to_list()
      event_dates = events_df["ds"] |> Series.to_list()

      Enum.zip(event_names, event_dates)
      |> Enum.group_by(fn {name, _date} -> name end, fn {_name, date} -> date end)
    end
  end

  defp build_feature_column(dates_list, event_dates, offset) do
    # For each date in dates_list, check if (date - offset) matches any event date
    # Equivalently: check if any (event_date + offset) matches date
    shifted_event_dates =
      event_dates
      |> Enum.map(fn event_date -> Date.add(event_date, offset) end)
      |> MapSet.new()

    dates_list
    |> Enum.map(fn date ->
      if MapSet.member?(shifted_event_dates, date), do: 1.0, else: 0.0
    end)
    |> Nx.tensor()
  end
end
