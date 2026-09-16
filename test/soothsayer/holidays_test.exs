defmodule Soothsayer.HolidaysTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Holidays

  doctest Holidays

  defp config(overrides) do
    Map.merge(
      %{countries: ["US"], steps_before: 0, steps_after: 0, types: [:public], language: "en"},
      overrides
    )
  end

  describe "dates/2" do
    test "names the US public holidays in English" do
      dates = Holidays.dates(config(%{}), 2023..2023)

      assert dates["Independence Day"] == [~D[2023-07-04]]
      assert dates["Christmas Day"] == [~D[2023-12-25]]
      assert dates["Thanksgiving Day"] == [~D[2023-11-23]]
      refute Map.has_key?(dates, "Valentine's Day")
    end

    test "types add the other holiday kinds" do
      dates = Holidays.dates(config(%{types: [:public, :observance]}), 2023..2023)
      assert dates["Valentine's Day"] == [~D[2023-02-14]]
    end

    test "covers every year asked for and merges the same name across countries" do
      dates = Holidays.dates(config(%{countries: ["US", "GB"]}), 2022..2023)

      assert dates["Christmas Day"] == [~D[2022-12-25], ~D[2023-12-25]]
      assert Map.has_key?(dates, "Boxing Day")
    end

    test "a state in the code adds its own holidays" do
      national = Holidays.names(config(%{}), 2023..2023)
      california = Holidays.names(config(%{countries: ["US-CA"]}), 2023..2023)

      assert "César Chávez Day" in (california -- national)
      assert national == Enum.sort(national)
      assert Holidays.dates(config(%{countries: []}), 2023..2023) == %{}
    end
  end

  describe "normalize_config!/1" do
    test "wraps and normalizes codes, keeping subdivision codes as written" do
      assert Holidays.normalize_config!(config(%{countries: :us})).countries == ["US"]

      assert Holidays.normalize_config!(config(%{countries: ["us-ca", :gb, "GB"]})).countries == [
               "GB",
               "US-CA"
             ]

      assert Holidays.normalize_config!(config(%{countries: ["CK-Aitutaki"]})).countries == [
               "CK-Aitutaki"
             ]
    end

    test "rejects unknown codes with dayoff's message" do
      assert_raise ArgumentError, ~r/unknown country "NARNIA". Known: AD, AE/, fn ->
        Soothsayer.new(%{holidays: %{countries: [:narnia]}})
      end
    end

    test "rejects the old keys, bad windows and bad types" do
      assert_raise ArgumentError,
                   ~r/holidays.regions and holidays.include_informal are gone/,
                   fn ->
                     Soothsayer.new(%{holidays: %{countries: ["US"], regions: ["us_ca"]}})
                   end

      assert_raise ArgumentError, ~r/holidays.steps_before must be an integer >= 0/, fn ->
        Soothsayer.new(%{holidays: %{countries: ["US"], steps_before: -1}})
      end

      assert_raise ArgumentError, ~r/holidays.types must be a non-empty list/, fn ->
        Soothsayer.new(%{holidays: %{countries: ["US"], types: [:informal]}})
      end
    end
  end
end
