defmodule Soothsayer.Holidays do
  @moduledoc """
  Country holidays as events, the way NeuralProphet's `add_country_holidays`
  does it. The dates come from [dayoff](https://hex.pm/packages/dayoff),
  which ships the date-holidays dataset: 200+ countries with their states
  and regions.

  Every holiday of the configured countries becomes its own event, named in
  English by default ("Independence Day", "Christmas Day"), with the window
  shared by all of them:

      Soothsayer.new(%{
        holidays: %{countries: ["US", "BR"], steps_before: 1, steps_after: 1}
      })

  A state or region goes in the code the dayoff way, `"US-CA"` or
  `"DE-BY-A"`. `types` picks which dayoff holiday types count, public ones
  by default. The dates are generated for the years of the data at fit, and
  again for the years being predicted, so nothing has to be listed by hand.
  The same holiday name from two codes is one event, as in NeuralProphet.
  """

  @type config :: %{
          optional(:names) => list(String.t()),
          countries: list(String.t()),
          steps_before: non_neg_integer(),
          steps_after: non_neg_integer(),
          mode: :additive | :multiplicative,
          regularization: nil | number(),
          types: list(Dayoff.Holiday.type()),
          language: String.t()
        }

  @doc """
  The country codes dayoff knows, sorted. States and regions are listed by
  `Dayoff.states/1` and `Dayoff.regions/2`.
  """
  @spec supported() :: list(String.t())
  def supported, do: Dayoff.countries() |> Map.keys() |> Enum.sort()

  @doc """
  Validates the `holidays` config and normalizes `countries` to a sorted
  list of dayoff codes with the country part uppercased. Raises
  `ArgumentError` on anything else, including the codes dayoff doesn't know.
  """
  @spec normalize_config!(map()) :: config()
  def normalize_config!(%{countries: countries} = config) do
    for key <- [:regions, :include_informal], Map.has_key?(config, key) do
      raise ArgumentError,
            "holidays.regions and holidays.include_informal are gone: put the state in the " <>
              "country code (\"US-CA\") and pick holiday types with types: [:public, :observance]"
    end

    countries = countries |> List.wrap() |> Enum.map(&code!/1) |> Enum.uniq() |> Enum.sort()
    validate_windows!(config)

    unless is_list(config.types) and config.types != [] and
             Enum.all?(config.types, &(&1 in Dayoff.Holiday.types())) do
      raise ArgumentError,
            "holidays.types must be a non-empty list from #{inspect(Dayoff.Holiday.types())}, " <>
              "got #{inspect(config.types)}"
    end

    unless is_binary(config.language) do
      raise ArgumentError,
            "holidays.language must be a language code like \"en\", got #{inspect(config.language)}"
    end

    validate_effect!(config)

    config
    |> Map.put(:countries, countries)
    |> Map.put_new(:mode, :additive)
    |> Map.put_new(:regularization, nil)
  end

  def normalize_config!(config) do
    raise ArgumentError,
          "holidays must be a map with :countries, :steps_before and :steps_after, " <>
            "got #{inspect(config)}"
  end

  defp validate_effect!(config) do
    unless Map.get(config, :mode, :additive) in [:additive, :multiplicative] do
      raise ArgumentError,
            "holidays.mode must be :additive or :multiplicative, got #{inspect(config.mode)}"
    end

    regularization = Map.get(config, :regularization)

    unless is_nil(regularization) or (is_number(regularization) and regularization >= 0) do
      raise ArgumentError,
            "holidays.regularization must be nil or a number >= 0, got #{inspect(regularization)}"
    end
  end

  defp validate_windows!(config) do
    if Map.has_key?(config, :lower_window) or Map.has_key?(config, :upper_window) do
      raise ArgumentError,
            "holidays use steps_before and steps_after now, both counts >= 0 " <>
              "(lower_window: -2, upper_window: 1 becomes steps_before: 2, steps_after: 1)"
    end

    for key <- [:steps_before, :steps_after],
        value = Map.get(config, key),
        not (is_integer(value) and value >= 0) do
      raise ArgumentError, "holidays.#{key} must be an integer >= 0, got #{inspect(value)}"
    end
  end

  # "us-ca" becomes "US-CA": the country is uppercased and the state and
  # region are spelled the way dayoff has them (it also has codes like
  # "Aitutaki"). dayoff raises for codes it doesn't know.
  defp code!(code) when is_atom(code) and not is_nil(code),
    do: code |> Atom.to_string() |> code!()

  defp code!(code) when is_binary(code) do
    case String.split(code, "-") do
      [country] ->
        country = String.upcase(country)
        Dayoff.languages(country)
        country

      [country, state] ->
        country = String.upcase(country)
        Enum.join([country, state!(country, state)], "-")

      [country, state, region] ->
        country = String.upcase(country)
        state = state!(country, state)

        region =
          subdivision!(
            Dayoff.regions(country, state),
            region,
            "#{country}-#{state} has no region"
          )

        Enum.join([country, state, region], "-")

      _ ->
        raise ArgumentError,
              "holidays.countries must be dayoff codes like \"US\" or \"US-CA\", got #{inspect(code)}"
    end
  end

  defp code!(code) do
    raise ArgumentError,
          "holidays.countries must be dayoff codes like \"US\" or \"US-CA\", got #{inspect(code)}"
  end

  defp state!(country, state),
    do: subdivision!(Dayoff.states(country), state, "#{country} has no state or region")

  defp subdivision!(known, code, message) do
    Enum.find(Map.keys(known), &(String.downcase(&1) == String.downcase(code))) ||
      raise(
        ArgumentError,
        "#{message} #{inspect(code)}. Known: #{known |> Map.keys() |> Enum.sort() |> Enum.join(", ")}"
      )
  end

  @doc """
  The holiday dates of the configured countries for the given years, by
  holiday name. Substitute days carry their own name and are their own
  event.

  ## Examples

      iex> config = %{countries: ["US"], steps_before: 0, steps_after: 0, types: [:public], language: "en"}
      iex> Soothsayer.Holidays.dates(config, 2023..2023)["Independence Day"]
      [~D[2023-07-04]]

  """
  @spec dates(config(), Enumerable.t()) :: %{String.t() => list(Date.t())}
  def dates(%{countries: []}, _years), do: %{}

  def dates(config, years) do
    for code <- config.countries,
        year <- years,
        holiday <- Dayoff.holidays(code, year, types: config.types, language: config.language) do
      holiday
    end
    |> Enum.group_by(& &1.name, & &1.date)
    |> Map.new(fn {name, dates} -> {name, dates |> Enum.uniq() |> Enum.sort(Date)} end)
  end

  @doc """
  The holiday names of the configured countries for the given years, sorted.
  """
  @spec names(config(), Enumerable.t()) :: list(String.t())
  def names(config, years), do: config |> dates(years) |> Map.keys() |> Enum.sort()
end
