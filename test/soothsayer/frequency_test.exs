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

  describe "month ends" do
    test "month-end dates stay on the month-end grid in both directions" do
      month_ends = [~D[2023-01-31], ~D[2023-02-28], ~D[2023-03-31], ~D[2023-04-30]]

      assert Frequency.range(~D[2022-12-31], ~D[2023-04-30], {1, :month}) == month_ends
      assert Frequency.steps_between(~D[2023-02-28], ~D[2023-04-30], {1, :month}) == 2
      assert Frequency.shift(~N[2023-02-28 06:00:00], -1, {1, :month}) == ~N[2023-01-31 06:00:00]

      # A 28th that isn't a month end keeps its day like any other date
      assert Frequency.shift(~D[2023-03-28], 1, {1, :month}) == ~D[2023-04-28]
    end

    test "monthly data on month ends fits with auto-regression" do
      timestamps = Enum.map(0..47, &Date.end_of_month(Date.shift(~D[2019-01-01], month: &1)))
      y = Enum.map(1..48, &(100.0 + &1 + 5 * :math.sin(&1)))
      df = Explorer.DataFrame.new(%{"ds" => timestamps, "y" => y})

      model =
        Soothsayer.new(%{
          seasonality: %{weekly: %{enabled: false}},
          ar: %{enabled: true, lags: 3},
          trend: %{changepoints: 0},
          epochs: 2
        })

      fitted = Soothsayer.fit(model, df)
      assert fitted.config.frequency == {1, :month}

      next = Explorer.Series.from_list([~D[2023-01-31], ~D[2023-02-28]])
      assert Nx.shape(Soothsayer.predict(fitted, next)) == {2, 1}
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
