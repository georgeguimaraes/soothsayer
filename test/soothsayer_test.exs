defmodule SoothsayerTest do
  use ExUnit.Case
  doctest Soothsayer

  alias Explorer.DataFrame
  alias Explorer.Series

  describe "Soothsayer" do
    test "basic workflow - create, fit, and predict" do
      # Create sample data
      date_range = Date.range(~D[2023-01-01], ~D[2023-01-31])
      dates = Enum.to_list(date_range)
      values = Enum.map(1..31, fn x -> :math.sin(x / 5) * 10 + 50 + :rand.normal(0, 2) end)

      df =
        DataFrame.new(%{
          "date" => dates,
          "y" => values
        })

      # Create a new Soothsayer model
      model = Soothsayer.new()

      # Fit the model
      {fitted_model, metrics} = Soothsayer.fit(model, df, "D")

      # Assert that fitting returns the expected structure
      assert is_map(metrics)
      assert Map.has_key?(metrics, :mse)
      assert Map.has_key?(metrics, :mae)

      # Generate predictions for the next 7 days
      last_date = List.last(dates)
      future_dates = Date.range(Date.add(last_date, 1), Date.add(last_date, 7))
      future_df = DataFrame.new(%{"date" => Enum.to_list(future_dates)})

      forecast = Soothsayer.predict(fitted_model, future_df)

      # Assert that the forecast has the expected structure
      assert %DataFrame{} = forecast
      assert DataFrame.n_rows(forecast) == 7
      assert "date" in DataFrame.names(forecast)
      assert "yhat" in DataFrame.names(forecast)

      # Check that predictions are within a reasonable range
      yhat_series = DataFrame.pull(forecast, "yhat")
      assert Series.min(yhat_series) >= 0
      assert Series.max(yhat_series) <= 100

      # Check that all predicted values are the same (last observed value)
      last_observed_value = List.last(values)
      assert Enum.all?(Series.to_list(yhat_series), fn x -> x == last_observed_value end)

      # Check that the dates in the forecast are correct
      forecast_dates = DataFrame.pull(forecast, "date") |> Series.to_list()
      expected_dates = Enum.to_list(future_dates)
      assert forecast_dates == expected_dates
    end
  end
end
