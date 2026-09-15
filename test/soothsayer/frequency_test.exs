defmodule Soothsayer.FrequencyTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Frequency

  doctest Frequency
  doctest Soothsayer.Timestamp

  describe "infer/1" do
    test "takes the most common gap, so one missing row doesn't change the answer" do
      hourly_with_gap = [
        ~N[2023-01-01 00:00:00],
        ~N[2023-01-01 01:00:00],
        ~N[2023-01-01 02:00:00],
        ~N[2023-01-01 04:00:00],
        ~N[2023-01-01 05:00:00]
      ]

      assert Frequency.infer(hourly_with_gap) == {1, :hour}
    end

    test "monthly dates infer a month despite the varying number of days" do
      dates = Enum.map(0..11, &Date.shift(~D[2023-01-01], month: &1))
      assert Frequency.infer(dates) == {1, :month}
    end

    test "raises on a step that isn't whole minutes" do
      assert_raise ArgumentError, ~r/Cannot infer a frequency/, fn ->
        Frequency.infer([~N[2023-01-01 00:00:00], ~N[2023-01-01 00:00:30]])
      end
    end
  end

  describe "steps_between/3" do
    test "raises when the timestamp is off the grid" do
      assert_raise ArgumentError, ~r/not a whole number of 5 minute steps/, fn ->
        Frequency.steps_between(~N[2023-01-01 00:00:00], ~N[2023-01-01 00:07:00], {5, :minute})
      end

      assert_raise ArgumentError, ~r/not a whole number of 1 month steps/, fn ->
        Frequency.steps_between(~D[2023-01-01], ~D[2023-02-15], {1, :month})
      end
    end
  end

  describe "validate!/1" do
    test "rejects malformed frequencies through the model config" do
      assert_raise ArgumentError, ~r/frequency must be :auto or \{amount, unit\}/, fn ->
        Soothsayer.new(%{frequency: {0, :hour}})
      end

      assert_raise ArgumentError, ~r/frequency must be :auto/, fn ->
        Soothsayer.new(%{frequency: :hourly})
      end

      assert %Soothsayer.Model{config: %{frequency: {15, :minute}}} =
               Soothsayer.new(%{frequency: {15, :minute}})
    end
  end
end
