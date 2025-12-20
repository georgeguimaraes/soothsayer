defmodule SoothsayerTest do
  use ExUnit.Case

  alias Explorer.DataFrame
  alias Explorer.Series

  describe "Soothsayer predictions" do
    test "trend-only prediction" do
      # Generate sample data with only trend and noise
      start_date = ~D[2019-01-01]
      end_date = ~D[2023-12-31]
      dates = Date.range(start_date, end_date)

      y =
        Enum.map(dates, fn date ->
          days_since_start = Date.diff(date, start_date)
          trend = 1000 + 0.5 * days_since_start
          # Add random noise
          noise = :rand.normal(0, 50)
          trend + noise
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      # Create and fit the model with only trend enabled and 10 epochs
      model =
        Soothsayer.new(%{
          trend: %{enabled: true},
          seasonality: %{
            yearly: %{enabled: false},
            weekly: %{enabled: false}
          },
          epochs: 10
        })

      fitted_model = Soothsayer.fit(model, df)

      # Make predictions for the next 30 days
      future_start = Date.add(end_date, 1)
      future_end = Date.add(future_start, 29)
      future_dates = Date.range(future_start, future_end)
      x_test = Series.from_list(Enum.to_list(future_dates))
      predictions = Soothsayer.predict(fitted_model, x_test) |> Nx.to_flat_list()

      # Check if predictions follow the trend (with tolerance for noise)
      Enum.zip(predictions, future_dates)
      |> Enum.each(fn {pred, date} ->
        days_since_start = Date.diff(date, start_date)
        expected_trend = 1000 + 0.5 * days_since_start
        # Increased tolerance due to noise and fewer epochs
        assert_in_delta pred, expected_trend, 100
      end)
    end

    test "seasonality-only prediction" do
      # Generate sample data with only seasonality and noise
      start_date = ~D[2019-01-01]
      end_date = ~D[2023-12-31]
      dates = Date.range(start_date, end_date)

      y =
        Enum.map(dates, fn date ->
          days_since_start = Date.diff(date, start_date)
          yearly_seasonality = 50 * :math.sin(2 * :math.pi() * days_since_start / 365.25)
          weekly_seasonality = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
          # Add random noise
          noise = :rand.normal(0, 10)
          yearly_seasonality + weekly_seasonality + noise
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      # Create and fit the model with only seasonality enabled and 10 epochs
      model =
        Soothsayer.new(%{
          trend: %{enabled: false},
          seasonality: %{
            yearly: %{enabled: true, fourier_terms: 3},
            weekly: %{enabled: true, fourier_terms: 3}
          },
          epochs: 10
        })

      fitted_model = Soothsayer.fit(model, df)

      # Make predictions for the next 30 days
      future_start = Date.add(end_date, 1)
      future_end = Date.add(future_start, 29)
      future_dates = Date.range(future_start, future_end)
      x_test = Series.from_list(Enum.to_list(future_dates))
      predictions = Soothsayer.predict(fitted_model, x_test) |> Nx.to_flat_list()

      # Check if predictions follow the seasonality pattern (with tolerance for noise)
      Enum.zip(predictions, future_dates)
      |> Enum.each(fn {pred, date} ->
        days_since_start = Date.diff(date, start_date)
        expected_yearly = 50 * :math.sin(2 * :math.pi() * days_since_start / 365.25)
        expected_weekly = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
        expected = expected_yearly + expected_weekly
        # Increased tolerance due to noise, approximation, and fewer epochs
        assert_in_delta pred, expected, 40
      end)
    end

    test "combined trend and seasonality prediction" do
      # Generate sample data with trend, seasonality, and noise
      start_date = ~D[2019-01-01]
      end_date = ~D[2023-12-31]
      dates = Date.range(start_date, end_date)

      y =
        Enum.map(dates, fn date ->
          days_since_start = Date.diff(date, start_date)
          trend = 1000 + 0.5 * days_since_start
          yearly_seasonality = 50 * :math.sin(2 * :math.pi() * days_since_start / 365.25)
          weekly_seasonality = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
          # Add random noise
          noise = :rand.normal(0, 20)
          trend + yearly_seasonality + weekly_seasonality + noise
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      # Create and fit the model with both trend and seasonality enabled
      model =
        Soothsayer.new(%{
          trend: %{enabled: true},
          seasonality: %{
            yearly: %{enabled: true, fourier_terms: 3},
            weekly: %{enabled: true, fourier_terms: 3}
          },
          epochs: 10
        })

      fitted_model = Soothsayer.fit(model, df)

      # Make predictions for the next 30 days
      future_start = Date.add(end_date, 1)
      future_end = Date.add(future_start, 29)
      future_dates = Date.range(future_start, future_end)
      x_test = Series.from_list(Enum.to_list(future_dates))
      predictions = Soothsayer.predict(fitted_model, x_test) |> Nx.to_flat_list()

      # Check if predictions follow the trend and seasonality (with tolerance for noise)
      Enum.zip(predictions, future_dates)
      |> Enum.each(fn {pred, date} ->
        days_since_start = Date.diff(date, start_date)
        expected_trend = 1000 + 0.5 * days_since_start
        expected_yearly = 50 * :math.sin(2 * :math.pi() * days_since_start / 365.25)
        expected_weekly = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
        expected = expected_trend + expected_yearly + expected_weekly
        # Increased tolerance due to noise, approximation, and fewer epochs
        assert_in_delta pred, expected, 100
      end)
    end
  end

  describe "Events predictions" do
    test "fit and predict with events" do
      # Generate sample data with trend and event spikes
      start_date = ~D[2023-01-01]
      end_date = ~D[2023-12-31]
      dates = Date.range(start_date, end_date)

      # Event dates: two "sale" events
      sale_dates = [~D[2023-03-15], ~D[2023-09-15]]

      y =
        Enum.map(dates, fn date ->
          days_since_start = Date.diff(date, start_date)
          trend = 100 + 0.1 * days_since_start
          # Add spike for sale events
          spike = if date in sale_dates, do: 50, else: 0
          # Add noise
          noise = :rand.normal(0, 5)
          trend + spike + noise
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      events_df = DataFrame.new(%{
        "event" => ["sale", "sale"],
        "ds" => sale_dates
      })

      # Create model with events
      model =
        Soothsayer.new(%{
          trend: %{enabled: true, n_changepoints: 0},
          seasonality: %{
            yearly: %{enabled: false},
            weekly: %{enabled: false}
          },
          events: %{
            "sale" => %{lower_window: 0, upper_window: 0}
          },
          epochs: 10
        })

      # Fit with events
      fitted_model = Soothsayer.fit(model, df, events: events_df)

      # Create future dates with a new sale event
      future_start = Date.add(end_date, 1)
      future_end = Date.add(future_start, 29)
      future_dates = Date.range(future_start, future_end)
      future_sale_date = ~D[2024-01-15]

      future_events_df = DataFrame.new(%{
        "event" => ["sale"],
        "ds" => [future_sale_date]
      })

      x_test = Series.from_list(Enum.to_list(future_dates))
      predictions = Soothsayer.predict(fitted_model, x_test, events: future_events_df)

      assert is_struct(predictions, Nx.Tensor)
      assert Nx.shape(predictions) == {30, 1}
    end

    test "predict_components returns events component" do
      start_date = ~D[2023-01-01]
      end_date = ~D[2023-06-30]
      dates = Date.range(start_date, end_date)

      sale_dates = [~D[2023-03-15]]

      y =
        Enum.map(dates, fn date ->
          trend = 100
          spike = if date in sale_dates, do: 50, else: 0
          trend + spike + :rand.normal(0, 2)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      events_df = DataFrame.new(%{
        "event" => ["sale"],
        "ds" => sale_dates
      })

      model =
        Soothsayer.new(%{
          trend: %{enabled: true, n_changepoints: 0},
          seasonality: %{
            yearly: %{enabled: false},
            weekly: %{enabled: false}
          },
          events: %{
            "sale" => %{lower_window: 0, upper_window: 0}
          },
          epochs: 5
        })

      fitted_model = Soothsayer.fit(model, df, events: events_df)

      # Predict on dates that include an event
      test_dates = Series.from_list([~D[2023-07-01], ~D[2023-07-15]])
      test_events = DataFrame.new(%{
        "event" => ["sale"],
        "ds" => [~D[2023-07-15]]
      })

      components = Soothsayer.predict_components(fitted_model, test_dates, events: test_events)

      assert Map.has_key?(components, :events)
      assert is_struct(components.events, Nx.Tensor)
      assert Nx.shape(components.events) == {2, 1}
    end

    test "events with windows" do
      start_date = ~D[2023-01-01]
      end_date = ~D[2023-03-31]
      dates = Date.range(start_date, end_date)

      # Event on Feb 15 with window -1 to +1 (affects Feb 14, 15, 16)
      event_date = ~D[2023-02-15]

      y =
        Enum.map(dates, fn date ->
          trend = 100
          # Add spike in the window around the event
          spike =
            cond do
              date == Date.add(event_date, -1) -> 20
              date == event_date -> 50
              date == Date.add(event_date, 1) -> 20
              true -> 0
            end
          trend + spike + :rand.normal(0, 2)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      events_df = DataFrame.new(%{
        "event" => ["promo"],
        "ds" => [event_date]
      })

      model =
        Soothsayer.new(%{
          trend: %{enabled: true, n_changepoints: 0},
          seasonality: %{
            yearly: %{enabled: false},
            weekly: %{enabled: false}
          },
          events: %{
            "promo" => %{lower_window: -1, upper_window: 1}
          },
          epochs: 5
        })

      fitted_model = Soothsayer.fit(model, df, events: events_df)

      # The model should have events configured with 3 features (window positions)
      assert map_size(fitted_model.config.events) > 0

      # Test prediction
      test_dates = Series.from_list([~D[2023-04-01], ~D[2023-04-02]])
      test_events = DataFrame.new(%{"event" => [], "ds" => []})

      predictions = Soothsayer.predict(fitted_model, test_dates, events: test_events)
      assert Nx.shape(predictions) == {2, 1}
    end

    test "get_event_effects returns learned coefficients" do
      start_date = ~D[2023-01-01]
      end_date = ~D[2023-06-30]
      dates = Date.range(start_date, end_date)

      # Event with known effect
      sale_dates = [~D[2023-03-15], ~D[2023-05-15]]

      y =
        Enum.map(dates, fn date ->
          trend = 100
          spike = if date in sale_dates, do: 50, else: 0
          trend + spike + :rand.normal(0, 2)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      events_df = DataFrame.new(%{
        "event" => ["sale", "sale"],
        "ds" => sale_dates
      })

      model =
        Soothsayer.new(%{
          trend: %{enabled: true, n_changepoints: 0},
          seasonality: %{
            yearly: %{enabled: false},
            weekly: %{enabled: false}
          },
          events: %{
            "sale" => %{lower_window: 0, upper_window: 0}
          },
          epochs: 10
        })

      fitted_model = Soothsayer.fit(model, df, events: events_df)

      effects = Soothsayer.get_event_effects(fitted_model)

      # Should have one key for the event
      assert Map.has_key?(effects, "sale_0")
      assert is_number(effects["sale_0"])
    end

    test "get_event_effects returns coefficients for windowed events" do
      start_date = ~D[2023-01-01]
      end_date = ~D[2023-03-31]
      dates = Date.range(start_date, end_date)

      event_date = ~D[2023-02-15]

      y =
        Enum.map(dates, fn date ->
          trend = 100
          spike =
            cond do
              date == Date.add(event_date, -1) -> 20
              date == event_date -> 50
              date == Date.add(event_date, 1) -> 20
              true -> 0
            end
          trend + spike + :rand.normal(0, 2)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      events_df = DataFrame.new(%{
        "event" => ["promo"],
        "ds" => [event_date]
      })

      model =
        Soothsayer.new(%{
          trend: %{enabled: true, n_changepoints: 0},
          seasonality: %{
            yearly: %{enabled: false},
            weekly: %{enabled: false}
          },
          events: %{
            "promo" => %{lower_window: -1, upper_window: 1}
          },
          epochs: 5
        })

      fitted_model = Soothsayer.fit(model, df, events: events_df)

      effects = Soothsayer.get_event_effects(fitted_model)

      # Should have three keys for the windowed event
      assert Map.has_key?(effects, "promo_-1")
      assert Map.has_key?(effects, "promo_0")
      assert Map.has_key?(effects, "promo_+1")
    end
  end
end
