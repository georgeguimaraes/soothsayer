defmodule SoothsayerTest do
  use ExUnit.Case
  doctest Soothsayer

  alias Explorer.DataFrame
  alias Explorer.Series

  describe "Soothsayer" do
    test "trend-based prediction with 5 years of data" do
      # Create 5 years of sample data with a linear trend and minimal noise
      start_date = ~D[2019-01-01]
      end_date = ~D[2023-12-31]
      dates = Date.range(start_date, end_date)

      y =
        Enum.with_index(dates, fn _, i ->
          # Linear growth: 100 units per year, starting at 1000
          days_since_start = Date.diff(Enum.at(dates, i), start_date)
          1000 + 100 / 365 * days_since_start + :rand.normal(0, 1)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      # Create and fit the model
      model = Soothsayer.new(%{trend: %{hidden_sizes: [64, 32]}})
      fitted_model = Soothsayer.fit(model, df)

      # Make predictions for the next 30 days
      future_start = Date.add(end_date, 1)
      future_end = Date.add(future_start, 29)
      future_dates = Date.range(future_start, future_end)
      x_test = Series.from_list(Enum.to_list(future_dates))
      predictions = Soothsayer.predict(fitted_model, x_test)

      # Convert predictions to a list
      predictions_list = Nx.to_flat_list(predictions)
      assert length(predictions_list) == 30

      # Check if predictions follow the linear trend (with small tolerance for noise)
      Enum.with_index(predictions_list, fn pred, i ->
        # 5 years plus prediction days
        days_since_start = 5 * 365 + i
        expected = 1000 + 100 / 365 * days_since_start
        assert_in_delta pred, expected, 5
      end)

      # Check if the predictions are monotonically increasing (trend should be preserved)
      assert Enum.zip(predictions_list, Enum.drop(predictions_list, 1))
             |> Enum.all?(fn {a, b} -> b > a end)

      # Print the first and last predictions for manual inspection
      IO.puts("First prediction: #{Enum.at(predictions_list, 0)}")
      IO.puts("Last prediction: #{Enum.at(predictions_list, -1)}")
    end
  end
end
