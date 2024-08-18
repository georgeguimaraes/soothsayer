defmodule SoothsayerTest do
  use ExUnit.Case
  doctest Soothsayer

  alias Explorer.DataFrame
  alias Explorer.Series

  describe "Soothsayer predictions" do
    test "trend-only prediction" do
      # Generate sample data with only trend
      start_date = ~D[2019-01-01]
      end_date = ~D[2023-12-31]
      dates = Date.range(start_date, end_date)

      y =
        Enum.map(dates, fn date ->
          days_since_start = Date.diff(date, start_date)
          1000 + 100 / 365 * days_since_start + :rand.normal(0, 5)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      # Create and fit the model with only trend enabled
      model =
        Soothsayer.new(%{
          trend_config: %{enabled: true, hidden_sizes: [64, 32]},
          seasonality_config: %{
            yearly: %{enabled: false},
            weekly: %{enabled: false}
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

      # Check if predictions follow the trend (with tolerance for noise)
      Enum.zip(predictions_list, future_dates)
      |> Enum.with_index()
      |> Enum.each(fn {{pred, date}, i} ->
        days_since_start = 5 * 365 + i
        expected_trend = 1000 + 100 / 365 * days_since_start
        assert_in_delta pred, expected_trend, 50
      end)
    end

    test "seasonality-only prediction" do
      # Generate sample data with only seasonality
      start_date = ~D[2019-01-01]
      end_date = ~D[2023-12-31]
      dates = Date.range(start_date, end_date)

      y =
        Enum.map(dates, fn date ->
          days_since_start = Date.diff(date, start_date)
          yearly_seasonality = 50 * :math.sin(2 * :math.pi() * days_since_start / 365.25)
          weekly_seasonality = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
          yearly_seasonality + weekly_seasonality + :rand.normal(0, 5)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      # Create and fit the model with only seasonality enabled
      model =
        Soothsayer.new(%{
          trend_config: %{enabled: false},
          seasonality_config: %{
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

      # Check if predictions follow the seasonality (with tolerance for noise)
      Enum.zip(predictions_list, future_dates)
      |> Enum.with_index()
      |> Enum.each(fn {{pred, date}, i} ->
        days_since_start = 5 * 365 + i
        expected_yearly_seasonality = 50 * :math.sin(2 * :math.pi() * days_since_start / 365.25)
        expected_weekly_seasonality = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
        expected = expected_yearly_seasonality + expected_weekly_seasonality
        assert_in_delta pred, expected, 50
      end)
    end

    test "combined trend and seasonality prediction" do
      # Generate sample data with trend and seasonality
      start_date = ~D[2019-01-01]
      end_date = ~D[2023-12-31]
      dates = Date.range(start_date, end_date)

      y =
        Enum.map(dates, fn date ->
          days_since_start = Date.diff(date, start_date)
          trend = 1000 + 100 / 365 * days_since_start
          yearly_seasonality = 50 * :math.sin(2 * :math.pi() * days_since_start / 365.25)
          weekly_seasonality = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
          trend + yearly_seasonality + weekly_seasonality + :rand.normal(0, 5)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      # Create and fit the model with both trend and seasonality enabled
      model =
        Soothsayer.new(%{
          trend_config: %{enabled: true, hidden_sizes: [64, 32]},
          seasonality_config: %{
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
        assert_in_delta pred, expected, 50
      end)
    end
  end
end
