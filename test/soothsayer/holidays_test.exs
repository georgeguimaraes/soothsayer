defmodule Soothsayer.HolidaysTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Holidays

  doctest Holidays

  defp config(overrides) do
    Map.merge(
      %{countries: [:us], steps_before: 0, steps_after: 0, regions: [], include_informal: false},
      overrides
    )
  end

  describe "dates/2" do
    test "names the US holidays in English on their rule dates" do
      dates = Holidays.dates(config(%{}), 2023..2023)

      assert dates["Independence Day"] == [~D[2023-07-04]]
      assert dates["Christmas Day"] == [~D[2023-12-25]]
      assert dates["Thanksgiving"] == [~D[2023-11-23]]
      refute Map.has_key?(dates, "Good Friday")
    end

    test "informal holidays come in with include_informal" do
      dates = Holidays.dates(config(%{include_informal: true}), 2023..2023)
      assert dates["Good Friday"] == [~D[2023-04-07]]
    end

    test "covers every year asked for and merges the same name across countries" do
      dates = Holidays.dates(config(%{countries: [:us, :gb]}), 2022..2023)

      assert dates["Christmas Day"] == [~D[2022-12-25], ~D[2023-12-25]]
      assert dates["Boxing Day"] == [~D[2022-12-26], ~D[2023-12-26]]
    end

    test "names are sorted and empty without countries" do
      names = Holidays.names(config(%{}), 2023..2023)
      assert names == Enum.sort(names)
      assert "Labor Day" in names
      assert Holidays.dates(config(%{countries: []}), 2023..2023) == %{}
    end
  end

  describe "normalize_config!/1" do
    test "wraps and normalizes country codes" do
      assert Holidays.normalize_config!(config(%{countries: "US"})).countries == [:us]

      assert Holidays.normalize_config!(config(%{countries: [:gb, "us", :gb]})).countries == [
               :gb,
               :us
             ]
    end

    test "rejects unknown countries, naming the supported ones" do
      assert_raise ArgumentError, ~r/got :narnia. Supported: at, au, be/, fn ->
        Soothsayer.new(%{holidays: %{countries: [:narnia]}})
      end
    end

    test "rejects bad windows" do
      assert_raise ArgumentError, ~r/holidays.steps_before must be an integer >= 0/, fn ->
        Soothsayer.new(%{holidays: %{countries: [:us], steps_before: -1}})
      end

      assert_raise ArgumentError, ~r/steps_before and steps_after now/, fn ->
        Soothsayer.new(%{holidays: %{countries: [:us], lower_window: -1}})
      end
    end
  end
end
