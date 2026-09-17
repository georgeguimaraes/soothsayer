defmodule Soothsayer.Events do
  @moduledoc """
  Event component for Soothsayer models.

  Handles network building and feature engineering for events (promotions,
  launches, holidays). Each event becomes one binary feature per window
  offset, 1.0 where the event is that many steps away.

  An event is `mode: :additive` (the default, its coefficients are added to
  the forecast) or `mode: :multiplicative` (its coefficients are fractions of
  the trend, so the effect grows with the level of the series, like
  NeuralProphet's multiplicative events). The feature columns are laid out
  additive events first, then multiplicative, each group sorted by name, so
  every mode is a contiguous slice of the input and gets its own linear
  layer: `events_dense` and `events_multiplicative_dense`.

  Where the dates come from, see `event_dates/3`: the events dataframe
  given to fit or predict, the occurrences the fitted model remembers from
  fit, yearly recurrence for events configured with `recurring: :yearly`,
  and the country holidays of `Soothsayer.Holidays`.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Frequency
  alias Soothsayer.Holidays
  alias Soothsayer.Layers
  alias Soothsayer.Timestamp

  # Network Building

  @doc """
  Creates the Axon input node for the events component.

  ## Parameters

    * `config` - Model configuration map with optional `:events` key.

  ## Returns

    An Axon input node `{nil, positions, n_features}` when events are
    configured (`positions` being the timestamps in a sample, see
    `Soothsayer.AR.positions/1`), `nil` otherwise.

  """
  @spec build_network_input(map()) :: Axon.t() | nil
  def build_network_input(%{events: events_config} = config) when map_size(events_config) > 0 do
    n = n_features(events_config)
    Axon.input("events", shape: {nil, AR.positions(config), n})
  end

  def build_network_input(_config), do: nil

  @doc """
  Builds the events component layer.

  ## Parameters

    * `input` - Axon input node from `build_network_input/1`.
    * `config` - Model configuration map.

  ## Returns

    A linear layer over every position (`{batch, positions}`) when events
    are configured, `Axon.constant(0)` otherwise.

  """
  @spec build_component(Axon.t() | nil, map()) :: %{
          additive: Axon.t() | nil,
          multiplicative: Axon.t() | nil
        }
  def build_component(nil, _config), do: %{additive: nil, multiplicative: nil}

  def build_component(input, %{events: events_config}) when map_size(events_config) > 0 do
    ranges = mode_ranges(events_config)

    %{
      additive: Layers.position_dense_over(input, ranges.additive, "events_dense"),
      multiplicative:
        Layers.position_dense_over(input, ranges.multiplicative, "events_multiplicative_dense")
    }
  end

  def build_component(_input, _config), do: %{additive: nil, multiplicative: nil}

  @doc """
  The column ranges of the events input by mode, `nil` for a mode with no
  columns. Additive columns come first.

  ## Examples

      iex> Events.mode_ranges(%{"a" => %{steps_before: 1, steps_after: 0}, "b" => %{steps_before: 0, steps_after: 0, mode: :multiplicative}})
      %{additive: 0..1, multiplicative: 2..2}

  """
  @spec mode_ranges(map()) :: %{additive: Range.t() | nil, multiplicative: Range.t() | nil}
  def mode_ranges(events_config) do
    {additive, multiplicative} = by_mode(events_config)
    additive_count = n_features(Map.new(additive))
    multiplicative_count = n_features(Map.new(multiplicative))

    %{
      additive: range_from(0, additive_count),
      multiplicative: range_from(additive_count, multiplicative_count)
    }
  end

  defp range_from(_start, 0), do: nil
  defp range_from(start, count), do: start..(start + count - 1)

  # The events split by mode, each group sorted by name. A spec without a
  # mode is additive, so configs built by hand keep working.
  defp by_mode(events_config) do
    events_config
    |> Enum.sort_by(fn {name, _} -> name end)
    |> Enum.split_with(fn {_name, spec} -> mode(spec) == :additive end)
  end

  @doc """
  The mode of an event spec, `:additive` unless it says `:multiplicative`.
  """
  @spec mode(map()) :: :additive | :multiplicative
  def mode(spec), do: Map.get(spec, :mode, :additive)

  @doc """
  The L1 lambda of every feature column, by layer, from each event's
  `regularization` (`nil` and `0` mean none). Same column order as
  `feature_names/1`, split by mode. Layers with no columns are left out.

  ## Examples

      iex> Events.regularization_weights(%{"a" => %{steps_before: 1, steps_after: 0, regularization: 0.5}, "b" => %{steps_before: 0, steps_after: 0}})
      %{"events_dense" => [0.5, 0.5, 0.0]}

  """
  @spec regularization_weights(map()) :: %{String.t() => list(float())}
  def regularization_weights(events_config) do
    {additive, multiplicative} = by_mode(events_config)

    [{"events_dense", additive}, {"events_multiplicative_dense", multiplicative}]
    |> Enum.reject(fn {_layer, events} -> events == [] end)
    |> Map.new(fn {layer, events} ->
      {layer,
       Enum.flat_map(events, fn {_name, spec} ->
         lambda = (Map.get(spec, :regularization) || 0) * 1.0
         List.duplicate(lambda, Range.size(offsets(spec)))
       end)}
    end)
  end

  # Weight Extraction

  @doc """
  Extracts the learned event coefficients from a fitted model.

  Returns a map of event feature names to their learned coefficients.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct with events configured.

  ## Returns

    A map of feature names to coefficient values.

  ## Examples

      iex> effects = Events.get_effects(fitted_model)
      %{"sale_0" => 45.2, "promo_-1" => 12.5}

  """
  @spec get_effects(Soothsayer.Model.t()) :: %{String.t() => float()}
  def get_effects(%Soothsayer.Model{} = model) do
    events_config = model.config[:events] || %{}

    if map_size(events_config) == 0 do
      raise ArgumentError, "No events configured on this model"
    end

    unless model.params do
      raise ArgumentError, "Model has not been fitted yet"
    end

    coefficients =
      Enum.flat_map(["events_dense", "events_multiplicative_dense"], fn layer ->
        case model.params.data[layer] do
          nil -> []
          %{"kernel" => kernel} -> kernel |> Nx.flatten() |> Nx.to_flat_list()
        end
      end)

    if coefficients == [] do
      raise ArgumentError, "Events layer not found in model params"
    end

    feature_names(events_config)
    |> Enum.zip(coefficients)
    |> Enum.into(%{})
  end

  # Feature Engineering

  @doc """
  Computes total number of event features based on config.

  Each event with `steps_before` and `steps_after` creates
  `steps_before + 1 + steps_after` features.

  ## Examples

      iex> Events.n_features(%{})
      0

      iex> Events.n_features(%{"sale" => %{steps_before: 0, steps_after: 0}})
      1

      iex> Events.n_features(%{"black_friday" => %{steps_before: 2, steps_after: 1}})
      4

  """
  @spec n_features(map()) :: non_neg_integer()
  def n_features(events_config) when events_config == %{}, do: 0

  def n_features(events_config) do
    events_config
    |> Enum.map(fn {_name, %{steps_before: before, steps_after: after_}} ->
      before + 1 + after_
    end)
    |> Enum.sum()
  end

  @doc """
  Returns list of feature names for all configured events, in the column
  order of the events input: additive events first, then multiplicative,
  each sorted by name.

  Names are formatted as "event_name_offset" where offset indicates
  the window position relative to the event date.

  ## Examples

      iex> Events.feature_names(%{})
      []

      iex> Events.feature_names(%{"sale" => %{steps_before: 0, steps_after: 0}})
      ["sale_0"]

      iex> Events.feature_names(%{"bf" => %{steps_before: 1, steps_after: 1}})
      ["bf_-1", "bf_0", "bf_+1"]

  """
  @spec feature_names(map()) :: [String.t()]
  def feature_names(events_config) when events_config == %{}, do: []

  def feature_names(events_config) do
    {additive, multiplicative} = by_mode(events_config)
    Enum.flat_map(additive ++ multiplicative, &feature_names_for_event/1)
  end

  defp feature_names_for_event({name, spec}) do
    Enum.map(offsets(spec), fn offset -> format_feature_name(name, offset) end)
  end

  defp offsets(%{steps_before: before, steps_after: after_}), do: -before..after_//1

  defp format_feature_name(name, offset) do
    offset_str = if offset > 0, do: "+#{offset}", else: "#{offset}"
    "#{name}_#{offset_str}"
  end

  @doc """
  Every occurrence of every configured event, by event name, for the
  timestamps about to be featurized.

  Takes the union of the occurrences the fitted model `remembered` for the
  series (as `frame_dates/1` gave them at fit) and the ones in `events_df`
  (`nil` for none), repeats the events configured with `recurring: :yearly`
  on their month and day over the years of `timestamps` (February 29 only
  in leap years), and adds the country holidays named in `config.holidays`
  for those years.

  ## Examples

      iex> config = %{events: %{"launch" => %{steps_before: 0, steps_after: 0, recurring: :yearly}}}
      iex> events_df = Explorer.DataFrame.new(%{"event" => ["launch"], "ds" => [~D[2022-03-01]]})
      iex> timestamps = [~N[2022-01-01 00:00:00], ~N[2023-12-31 00:00:00]]
      iex> Soothsayer.Events.event_dates(events_df, %{}, config, timestamps)
      %{"launch" => [~N[2022-03-01 00:00:00], ~N[2023-03-01 00:00:00]]}

  """
  @spec event_dates(
          DataFrame.t() | nil,
          %{String.t() => list(Timestamp.t())},
          map(),
          list(Timestamp.t())
        ) :: %{String.t() => list(Timestamp.t())}
  def event_dates(events_df, remembered, config, timestamps) do
    events_config = config[:events] || %{}
    from_frame = frame_dates(events_df)
    years = years(timestamps)

    given =
      remembered
      |> Map.merge(from_frame, fn _name, older, newer -> older ++ newer end)
      |> Map.take(Map.keys(events_config))

    recurring =
      for {name, %{recurring: :yearly}} <- events_config, into: %{} do
        {name, Enum.flat_map(Map.get(given, name, []), &repeat_yearly(&1, years))}
      end

    holidays =
      for name <- get_in(config, [:holidays, :names]) || [],
          dates = Holidays.dates(config.holidays, years),
          into: %{} do
        {name, Enum.map(Map.get(dates, name, []), &Timestamp.to_naive_datetime/1)}
      end

    [given, recurring, holidays]
    |> Enum.reduce(&Map.merge(&1, &2, fn _name, left, right -> left ++ right end))
    |> Map.new(fn {name, dates} -> {name, dates |> Enum.uniq() |> Enum.sort(NaiveDateTime)} end)
  end

  @doc """
  The occurrences in an events dataframe by event name, as naive datetimes.
  `nil` or an empty dataframe give an empty map.
  """
  @spec frame_dates(DataFrame.t() | nil) :: %{String.t() => list(Timestamp.t())}
  def frame_dates(nil), do: %{}
  def frame_dates(%DataFrame{} = events_df), do: build_event_dates_map(events_df)

  @doc """
  The range of years the timestamps span, an empty range when there are none.
  """
  @spec years(list(Timestamp.input())) :: Range.t()
  def years([]), do: 0..-1//1

  def years(timestamps) do
    {first, last} = Enum.min_max_by(timestamps, & &1.year)
    first.year..last.year//1
  end

  defp repeat_yearly(timestamp, years) do
    naive = Timestamp.to_naive_datetime(timestamp)

    for year <- years,
        {:ok, date} <- [Date.new(year, naive.month, naive.day)],
        do: NaiveDateTime.new!(date, NaiveDateTime.to_time(naive))
  end

  @doc """
  Builds event features tensor from timestamps and the event dates.

  For each timestamp, creates binary features indicating whether each event
  (with its window offsets) occurs then. Window offsets are steps of
  `frequency`, so on daily data a window of `-1..1` covers the day before
  and after, and on hourly data the hour before and after. Event dates
  given as plain dates mean midnight.

  ## Parameters

    * `dates` - An Explorer Series of dates or naive datetimes.
    * `event_dates` - A DataFrame with "event" and "ds" columns, or a map
      from event name to occurrences as built by `event_dates/3`.
    * `events_config` - Map of event configurations.
    * `frequency` - The data frequency, `{1, :day}` by default.

  ## Returns

    A tensor of shape {n_dates, n_event_features} with 1.0 where
    events occur and 0.0 otherwise.

  ## Examples

      iex> dates = Explorer.Series.from_list([~D[2023-01-01], ~D[2023-01-02]])
      iex> events_df = Explorer.DataFrame.new(%{"event" => ["sale"], "ds" => [~D[2023-01-02]]})
      iex> config = %{"sale" => %{steps_before: 0, steps_after: 0}}
      iex> Events.build_features(dates, events_df, config)
      #Nx.Tensor<
        f32[2][1]
        [
          [0.0],
          [1.0]
        ]
      >

  """
  @spec build_features(Series.t(), DataFrame.t() | map(), map(), Frequency.t()) ::
          Nx.Tensor.t() | nil
  def build_features(dates, event_dates, events_config, frequency \\ {1, :day})

  def build_features(_dates, _event_dates, events_config, _frequency) when events_config == %{} do
    nil
  end

  def build_features(dates, %DataFrame{} = events_df, events_config, frequency) do
    build_features(dates, build_event_dates_map(events_df), events_config, frequency)
  end

  def build_features(dates, event_dates_map, events_config, frequency) do
    dates_list = Timestamp.from_series(dates)

    # For each event (sorted by name) and each window position, create a column
    columns =
      events_config
      |> Enum.sort_by(fn {name, _} -> name end)
      |> Enum.flat_map(fn {event_name, spec} ->
        event_dates = Map.get(event_dates_map, event_name, [])

        spec
        |> offsets()
        |> Enum.map(fn offset ->
          build_feature_column(dates_list, event_dates, offset, frequency)
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
      event_dates = Timestamp.from_series(events_df["ds"])

      Enum.zip(event_names, event_dates)
      |> Enum.group_by(fn {name, _date} -> name end, fn {_name, date} -> date end)
    end
  end

  defp build_feature_column(dates_list, event_dates, offset, frequency) do
    # For each date in dates_list, check if (date - offset) matches any event date
    # Equivalently: check if any (event_date + offset) matches date
    shifted_event_dates =
      event_dates
      |> Enum.map(fn event_date -> Frequency.shift(event_date, offset, frequency) end)
      |> MapSet.new()

    dates_list
    |> Enum.map(fn date ->
      if MapSet.member?(shifted_event_dates, date), do: 1.0, else: 0.0
    end)
    |> Nx.tensor()
  end
end
