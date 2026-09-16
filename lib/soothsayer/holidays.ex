defmodule Soothsayer.Holidays do
  @moduledoc """
  Country holidays as events, the way NeuralProphet's `add_country_holidays`
  does it.

  Dates come from the `holidefs` package, an optional dependency:

      {:holidefs, "~> 0.4"}

  Every holiday of the configured countries becomes its own event, named as
  holidefs names it in English ("Independence Day", "Christmas Day"), with
  the window shared by all of them:

      Soothsayer.new(%{
        holidays: %{countries: [:us], steps_before: 1, steps_after: 1}
      })

  The dates are generated for the years of the data at fit, and again for
  the years being predicted, so nothing has to be listed by hand. The same
  holiday name from two countries is one event, as in NeuralProphet.
  """

  @compile {:no_warn_undefined, [Holidefs, Gettext]}

  @type config :: %{
          optional(:names) => list(String.t()),
          countries: list(atom()),
          steps_before: non_neg_integer(),
          steps_after: non_neg_integer(),
          regions: list(String.t()),
          include_informal: boolean()
        }

  @doc """
  Whether the holidefs package is available.
  """
  @spec available?() :: boolean()
  def available?, do: Code.ensure_loaded?(Holidefs)

  @doc """
  The locale codes holidefs knows, sorted.
  """
  @spec supported() :: list(atom())
  def supported, do: Holidefs.locales() |> Map.keys() |> Enum.sort()

  @doc """
  Validates the `holidays` config and normalizes `countries` to a sorted
  list of locale atoms. Raises `ArgumentError` on anything else.
  """
  @spec normalize_config!(map()) :: config()
  def normalize_config!(%{countries: countries} = config) do
    countries = List.wrap(countries)

    if countries != [] and not available?() do
      raise ArgumentError,
            "Country holidays need the holidefs package. " <>
              "Add {:holidefs, \"~> 0.4\"} to your deps."
    end

    countries = countries |> Enum.map(&locale!/1) |> Enum.uniq() |> Enum.sort()

    validate_windows!(config)

    unless is_list(config.regions) and Enum.all?(config.regions, &is_binary/1) do
      raise ArgumentError,
            "holidays.regions must be a list of holidefs region strings like \"us_ca\", " <>
              "got #{inspect(config.regions)}"
    end

    unless is_boolean(config.include_informal) do
      raise ArgumentError,
            "holidays.include_informal must be true or false, got #{inspect(config.include_informal)}"
    end

    %{config | countries: countries}
  end

  def normalize_config!(config) do
    raise ArgumentError,
          "holidays must be a map with :countries, :steps_before and :steps_after, " <>
            "got #{inspect(config)}"
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

  defp locale!(country) when is_atom(country) do
    if country in supported() do
      country
    else
      raise ArgumentError,
            "holidays.countries must be holidefs locale codes, got #{inspect(country)}. " <>
              "Supported: #{Enum.map_join(supported(), ", ", &Atom.to_string/1)}"
    end
  end

  defp locale!(country) when is_binary(country) do
    supported()
    |> Enum.find(&(Atom.to_string(&1) == String.downcase(country)))
    |> case do
      nil -> locale!(String.to_atom(country))
      locale -> locale
    end
  end

  defp locale!(country) do
    raise ArgumentError,
          "holidays.countries must be holidefs locale codes like :us, got #{inspect(country)}"
  end

  @doc """
  The holiday dates of the configured countries for the given years, by
  holiday name.

  Names are the English ones holidefs returns, whatever Gettext locale the
  calling process has set, so they are stable feature keys. Dates are the
  rule dates (not the observed ones), sorted and unique per name.

  ## Examples

      iex> config = %{countries: [:us], steps_before: 0, steps_after: 0, regions: [], include_informal: false}
      iex> Soothsayer.Holidays.dates(config, 2023..2023)["Independence Day"]
      [~D[2023-07-04]]

  """
  @spec dates(config(), Enumerable.t()) :: %{String.t() => list(Date.t())}
  def dates(%{countries: []}, _years), do: %{}

  def dates(config, years) do
    options = [regions: config.regions, include_informal?: config.include_informal]

    holidays =
      Gettext.with_locale(Holidefs.Gettext, "en", fn ->
        for country <- config.countries,
            year <- years,
            holiday <- year!(country, year, options) do
          holiday
        end
      end)

    holidays
    |> Enum.group_by(& &1.name, & &1.date)
    |> Map.new(fn {name, dates} -> {name, dates |> Enum.uniq() |> Enum.sort(Date)} end)
  end

  @doc """
  The holiday names of the configured countries for the given years, sorted.
  """
  @spec names(config(), Enumerable.t()) :: list(String.t())
  def names(config, years), do: config |> dates(years) |> Map.keys() |> Enum.sort()

  defp year!(country, year, options) do
    case Holidefs.year(country, year, options) do
      {:ok, holidays} ->
        holidays

      {:error, reason} ->
        raise ArgumentError,
              "holidefs has no holidays for #{inspect(country)} in #{year}: #{inspect(reason)}"
    end
  end
end
