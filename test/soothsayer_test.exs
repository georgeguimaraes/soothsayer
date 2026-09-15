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

      events_df =
        DataFrame.new(%{
          "event" => ["sale", "sale"],
          "ds" => sale_dates
        })

      # Create model with events
      model =
        Soothsayer.new(%{
          trend: %{enabled: true, changepoints: 0},
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

      future_events_df =
        DataFrame.new(%{
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

      events_df =
        DataFrame.new(%{
          "event" => ["sale"],
          "ds" => sale_dates
        })

      model =
        Soothsayer.new(%{
          trend: %{enabled: true, changepoints: 0},
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

      test_events =
        DataFrame.new(%{
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

      events_df =
        DataFrame.new(%{
          "event" => ["promo"],
          "ds" => [event_date]
        })

      model =
        Soothsayer.new(%{
          trend: %{enabled: true, changepoints: 0},
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

      events_df =
        DataFrame.new(%{
          "event" => ["sale", "sale"],
          "ds" => sale_dates
        })

      model =
        Soothsayer.new(%{
          trend: %{enabled: true, changepoints: 0},
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

      events_df =
        DataFrame.new(%{
          "event" => ["promo"],
          "ds" => [event_date]
        })

      model =
        Soothsayer.new(%{
          trend: %{enabled: true, changepoints: 0},
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

  describe "component decomposition" do
    test "components sum to combined and disabled components are zero" do
      start_date = ~D[2020-01-01]
      end_date = ~D[2022-12-31]
      dates = Date.range(start_date, end_date)
      sale_dates = [~D[2021-06-15], ~D[2022-06-15]]

      y =
        Enum.map(dates, fn date ->
          days = Date.diff(date, start_date)
          trend = 100 + 0.2 * days
          yearly = 15 * :math.sin(2 * :math.pi() * days / 365.25)
          weekly = 5 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
          spike = if date in sale_dates, do: 40, else: 0
          trend + yearly + weekly + spike + :rand.normal(0, 2)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})
      events_df = DataFrame.new(%{"event" => ["sale", "sale"], "ds" => sale_dates})

      model =
        Soothsayer.new(%{
          events: %{"sale" => %{lower_window: 0, upper_window: 0}},
          epochs: 5
        })

      fitted_model = Soothsayer.fit(model, df, events: events_df)

      future_dates = Date.range(~D[2023-01-01], ~D[2023-01-31]) |> Enum.to_list()
      future_events = DataFrame.new(%{"event" => ["sale"], "ds" => [~D[2023-01-15]]})

      components =
        Soothsayer.predict_components(fitted_model, Series.from_list(future_dates),
          events: future_events
        )

      summed =
        [
          :trend,
          :yearly_seasonality,
          :weekly_seasonality,
          :daily_seasonality,
          :ar,
          :events,
          :regressors,
          :lagged_regressors
        ]
        |> Enum.map(&components[&1])
        |> Enum.reduce(&Nx.add/2)

      assert Nx.all_close(summed, components.combined, atol: 1.0e-2) |> Nx.to_number() == 1

      # AR is disabled by default, so its component must be exactly zero
      assert components.ar |> Nx.abs() |> Nx.sum() |> Nx.to_number() == 0.0

      # Trend carries the level, so it should sit near the actual series values
      trend_values = Nx.to_flat_list(components.trend)
      assert Enum.all?(trend_values, fn v -> v > 150 and v < 400 end)
    end
  end

  describe "multiplicative seasonality" do
    # y = trend * (1 + 0.3 * yearly cycle): the seasonal swing grows with the level
    defp multiplicative_series(dates, start_date) do
      Enum.map(dates, fn date ->
        days = Date.diff(date, start_date)
        trend = 100 + 0.2 * days
        cycle = 0.3 * :math.sin(2 * :math.pi() * days / 365.25)
        trend * (1 + cycle) + :rand.normal(0, 2)
      end)
    end

    defp mode_config(mode) do
      %{
        trend: %{changepoints: 0},
        seasonality: %{mode: mode, weekly: %{enabled: false}},
        epochs: 60
      }
    end

    test "fits a series whose seasonal swing grows with the trend better than additive" do
      :rand.seed(:exsss, {5, 6, 7})
      start_date = ~D[2018-01-01]
      training_dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      holdout_dates = Date.range(~D[2023-01-01], ~D[2023-12-31]) |> Enum.to_list()

      df =
        DataFrame.new(%{
          "ds" => training_dates,
          "y" => multiplicative_series(training_dates, start_date)
        })

      holdout = multiplicative_series(holdout_dates, start_date)

      mean_absolute_error = fn mode ->
        fitted_model = Soothsayer.fit(Soothsayer.new(mode_config(mode)), df)
        predictions = Soothsayer.predict(fitted_model, Series.from_list(holdout_dates))

        predictions
        |> Nx.flatten()
        |> Nx.subtract(Nx.tensor(holdout))
        |> Nx.abs()
        |> Nx.mean()
        |> Nx.to_number()
      end

      additive_error = mean_absolute_error.(:additive)
      multiplicative_error = mean_absolute_error.(:multiplicative)

      assert multiplicative_error < additive_error * 0.7
    end

    test "components still sum to combined" do
      :rand.seed(:exsss, {5, 6, 7})
      start_date = ~D[2020-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      df = DataFrame.new(%{"ds" => dates, "y" => multiplicative_series(dates, start_date)})

      fitted_model =
        Soothsayer.fit(Soothsayer.new(%{mode_config(:multiplicative) | epochs: 5}), df)

      future_dates = Date.range(~D[2023-01-01], ~D[2023-03-31]) |> Enum.to_list()
      components = Soothsayer.predict_components(fitted_model, Series.from_list(future_dates))

      summed =
        [
          :trend,
          :yearly_seasonality,
          :weekly_seasonality,
          :daily_seasonality,
          :ar,
          :events,
          :regressors,
          :lagged_regressors
        ]
        |> Enum.map(&components[&1])
        |> Enum.reduce(&Nx.add/2)

      assert Nx.all_close(summed, components.combined, atol: 1.0e-2) |> Nx.to_number() == 1

      # The seasonal effect is not a constant offset, it moves with the trend
      assert components.yearly_seasonality |> Nx.abs() |> Nx.sum() |> Nx.to_number() > 0
    end

    test "rejects unknown modes" do
      assert_raise ArgumentError, ~r/seasonality.mode must be one of/, fn ->
        Soothsayer.new(%{seasonality: %{mode: :logarithmic}})
      end
    end
  end

  describe "sub-daily data" do
    defp hourly_frame(days) do
      timestamps =
        Enum.map(0..(days * 24 - 1), &NaiveDateTime.add(~N[2023-01-01 00:00:00], &1, :hour))

      y =
        Enum.map(timestamps, fn timestamp ->
          hour = timestamp.hour
          100 + 20 * :math.sin(2 * :math.pi() * hour / 24) + :rand.normal(0, 1)
        end)

      DataFrame.new(%{"ds" => timestamps, "y" => y})
    end

    test "infers the frequency, enables daily seasonality and forecasts by the hour" do
      :rand.seed(:exsss, {7, 7, 7})
      df = hourly_frame(10)

      model =
        Soothsayer.new(%{
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          ar: %{enabled: true, lags: 24, forecast_steps: 6},
          trend: %{changepoints: 0},
          epochs: 5,
          seed: 1
        })

      fitted = Soothsayer.fit(model, df)

      assert fitted.config.frequency == {1, :hour}
      assert fitted.config.seasonality.daily.enabled == true
      assert fitted.config.first_timestamp == ~N[2023-01-01 00:00:00]

      # The next day, requested as its 24 hourly timestamps
      next_day = Enum.map(1..24, &NaiveDateTime.add(~N[2023-01-10 23:00:00], &1, :hour))
      components = Soothsayer.predict_components(fitted, Series.from_list(next_day))

      assert Nx.shape(components.combined) == {24, 1}
      assert components.daily_seasonality |> Nx.abs() |> Nx.sum() |> Nx.to_number() > 0

      # A plain date means midnight and sits on the hourly grid
      assert Nx.shape(Soothsayer.predict(fitted, Series.from_list([~D[2023-01-11]]))) == {1, 1}

      assert_raise ArgumentError, ~r/not a whole number of 1 hour steps/, fn ->
        Soothsayer.predict(fitted, Series.from_list([~N[2023-01-11 00:30:00]]))
      end
    end

    test "date events line up with midnight on hourly data" do
      :rand.seed(:exsss, {8, 8, 8})
      df = hourly_frame(6)
      events_df = DataFrame.new(%{"event" => ["sale"], "ds" => [~D[2023-01-03]]})

      model =
        Soothsayer.new(%{
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          events: %{"sale" => %{lower_window: 0, upper_window: 0}},
          trend: %{changepoints: 0},
          epochs: 2
        })

      fitted = Soothsayer.fit(model, df, events: events_df)

      hours = Enum.map(0..3, &NaiveDateTime.add(~N[2023-01-02 23:00:00], &1, :hour))

      components =
        Soothsayer.predict_components(fitted, Series.from_list(hours), events: events_df)

      [before, midnight, after_one, after_two] = Nx.to_flat_list(components.events)

      # The event feature is z-scored, so hours without the event share one
      # baseline value and only midnight on the event day moves off it.
      assert before == after_one and after_one == after_two
      assert midnight != before
    end

    test "rejects unsorted timestamps and non-date ds columns" do
      model = Soothsayer.new(%{epochs: 1})

      unsorted =
        DataFrame.new(%{
          "ds" => [~D[2023-01-02], ~D[2023-01-01], ~D[2023-01-03]],
          "y" => [1.0, 2.0, 3.0]
        })

      assert_raise ArgumentError,
                   ~r/strictly increasing, found 2023-01-01 after 2023-01-02/,
                   fn ->
                     Soothsayer.fit(model, unsorted)
                   end

      strings = DataFrame.new(%{"ds" => ["2023-01-01", "2023-01-02"], "y" => [1.0, 2.0]})

      assert_raise ArgumentError, ~r/must be a date or naive datetime series/, fn ->
        Soothsayer.fit(model, strings)
      end
    end

    test "rejects missing target values" do
      model = Soothsayer.new(%{epochs: 1})
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04]]

      with_nil = DataFrame.new(%{"ds" => dates, "y" => [1.0, nil, 3.0, 4.0]})

      assert_raise ArgumentError, ~r/y column has 1 missing values/, fn ->
        Soothsayer.fit(model, with_nil)
      end

      with_nan = DataFrame.new(%{"ds" => dates, "y" => [1.0, :nan, :nan, 4.0]})

      assert_raise ArgumentError, ~r/y column has 2 missing values/, fn ->
        Soothsayer.fit(model, with_nan)
      end
    end

    test "rejects bad seasonality enabled values" do
      assert_raise ArgumentError,
                   ~r/seasonality.daily.enabled must be true, false or :auto/,
                   fn ->
                     Soothsayer.new(%{seasonality: %{daily: %{enabled: :always}}})
                   end
    end
  end

  describe "training defaults" do
    test "auto learning rate and epochs are resolved and recorded on the fitted model" do
      dates = Date.range(~D[2022-01-01], ~D[2022-12-31]) |> Enum.to_list()

      y =
        Enum.map(dates, fn date ->
          10 + 0.05 * Date.diff(date, ~D[2022-01-01]) + :rand.normal(0, 1)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      model = Soothsayer.new(%{seed: 1})
      assert model.config.learning_rate == :auto
      assert model.config.epochs == :auto
      assert model.config.schedule == :one_cycle

      fitted_model = Soothsayer.fit(model, df)

      assert is_float(fitted_model.config.learning_rate)

      assert fitted_model.config.learning_rate > 1.0e-5 and
               fitted_model.config.learning_rate < 1.0

      assert fitted_model.config.epochs == Soothsayer.Trainer.auto_epochs(365)
    end

    test "rejects unknown schedules and optimizers and bad rates" do
      assert_raise ArgumentError, ~r/schedule must be/, fn ->
        Soothsayer.new(%{schedule: :cyclic})
      end

      assert_raise ArgumentError, ~r/optimizer must be/, fn ->
        Soothsayer.new(%{optimizer: :sgd})
      end

      assert_raise ArgumentError, ~r/learning_rate must be/, fn ->
        Soothsayer.new(%{learning_rate: -1})
      end

      assert_raise ArgumentError, ~r/epochs must be/, fn -> Soothsayer.new(%{epochs: 0}) end
    end
  end
end
