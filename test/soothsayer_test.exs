defmodule SoothsayerTest do
  use ExUnit.Case
  doctest Soothsayer

  alias Explorer.DataFrame
  alias Explorer.Series

  describe "Soothsayer" do
    test "trend and seasonality prediction with 5 years of data" do
      # Create 5 years of sample data with a linear trend, yearly and weekly seasonality, and minimal noise
      start_date = ~D[2019-01-01]
      end_date = ~D[2023-12-31]
      dates = Date.range(start_date, end_date)

      y =
        Enum.map(dates, fn date ->
          # Linear growth: 100 units per year, starting at 1000
          days_since_start = Date.diff(date, start_date)
          trend = 1000 + 100 / 365 * days_since_start

          # Yearly seasonality
          yearly_seasonality = 50 * :math.sin(2 * :math.pi() * days_since_start / 365.25)

          # Weekly seasonality
          weekly_seasonality = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)

          trend + yearly_seasonality + weekly_seasonality + :rand.normal(0, 5)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      # Create and fit the model
      model =
        Soothsayer.new(%{
          trend: %{hidden_sizes: [64, 32]},
          seasonality: %{
            yearly: %{enabled: true, fourier_terms: 6},
            weekly: %{enabled: true, fourier_terms: 3}
          }
        })

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

      # Check if predictions follow the trend and seasonality (with tolerance for noise)
      Enum.zip(predictions_list, future_dates)
      |> Enum.with_index()
      |> Enum.each(fn {{pred, date}, i} ->
        days_since_start = 5 * 365 + i
        expected_trend = 1000 + 100 / 365 * days_since_start
        expected_yearly_seasonality = 50 * :math.sin(2 * :math.pi() * days_since_start / 365.25)
        expected_weekly_seasonality = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
        expected = expected_trend + expected_yearly_seasonality + expected_weekly_seasonality
        # Increased tolerance to account for model approximation
        assert_in_delta pred, expected, 30
      end)

      # Print the first and last predictions for manual inspection
      IO.puts("First prediction: #{Enum.at(predictions_list, 0)}")
      IO.puts("Last prediction: #{Enum.at(predictions_list, -1)}")
    end
  end
end
