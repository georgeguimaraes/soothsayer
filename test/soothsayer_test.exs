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
      predictions = Soothsayer.predict(fitted_model, x_test)["yhat"] |> Series.to_list()

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
      predictions = Soothsayer.predict(fitted_model, x_test)["yhat"] |> Series.to_list()

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
      predictions = Soothsayer.predict(fitted_model, x_test)["yhat"] |> Series.to_list()

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
            "sale" => %{steps_before: 0, steps_after: 0}
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

      assert is_struct(predictions, DataFrame)
      assert DataFrame.n_rows(predictions) == 30
      assert "events" in DataFrame.names(predictions)
    end

    defp holiday_frame(spike_on) do
      dates = Date.range(~D[2022-01-01], ~D[2023-12-31]) |> Enum.to_list()

      y =
        Enum.map(dates, fn date ->
          spike = if spike_on.(date), do: 50.0, else: 0.0
          100 + 0.05 * Date.diff(date, ~D[2022-01-01]) + spike + :rand.normal(0, 2)
        end)

      DataFrame.new(%{"ds" => dates, "y" => y})
    end

    defp event_model(overrides) do
      Soothsayer.new(
        Map.merge(
          %{
            trend: %{changepoints: 0},
            seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
            epochs: 15,
            learning_rate: 0.05,
            seed: 3
          },
          overrides
        )
      )
    end

    test "country holidays become events with their own effects, no dataframe needed" do
      :rand.seed(:exsss, {4, 4, 4})
      df = holiday_frame(&((&1.month == 7 and &1.day == 4) or (&1.month == 12 and &1.day == 25)))
      fitted = Soothsayer.fit(event_model(%{holidays: %{countries: [:us]}}), df)

      assert "Independence Day" in Map.keys(fitted.config.events)
      assert fitted.config.holidays.names == Enum.sort(fitted.config.holidays.names)

      us_public_names =
        for year <- 2022..2023,
            holiday <- Dayoff.holidays("US", year, types: [:public]),
            uniq: true,
            do: holiday.name

      assert length(fitted.config.holidays.names) == length(us_public_names)

      effects = Soothsayer.get_event_effects(fitted)

      strongest =
        effects
        |> Enum.sort_by(fn {_name, effect} -> -effect end)
        |> Enum.take(2)
        |> Enum.map(&elem(&1, 0))

      assert Enum.sort(strongest) == ["Christmas Day_0", "Independence Day_0"]

      # next year's holiday is known without an events dataframe
      around = Series.from_list([~D[2024-07-03], ~D[2024-07-04], ~D[2024-07-05]])

      [before, on_the_day, after_the_day] =
        Soothsayer.predict(fitted, around)["yhat"] |> Series.to_list()

      assert on_the_day > before + 20 and on_the_day > after_the_day + 20
    end

    test "a yearly recurring event given once applies to every year" do
      :rand.seed(:exsss, {5, 5, 5})
      df = holiday_frame(&(&1.month == 5 and &1.day == 10))
      events_df = DataFrame.new(%{"event" => ["founders_day"], "ds" => [~D[2022-05-10]]})

      model =
        event_model(%{
          events: %{"founders_day" => %{recurring: :yearly}}
        })

      fitted = Soothsayer.fit(model, df, events: events_df)

      # the 2023 occurrence was never listed, in-sample and next year alike
      in_sample = Soothsayer.predict(fitted, Series.from_list([~D[2023-05-09], ~D[2023-05-10]]))
      [before, on_the_day] = Series.to_list(in_sample["yhat"])
      assert on_the_day > before + 20

      next_year = Soothsayer.predict(fitted, Series.from_list([~D[2024-05-09], ~D[2024-05-10]]))
      [before, on_the_day] = Series.to_list(next_year["yhat"])
      assert on_the_day > before + 20
    end

    test "occurrences given at fit are remembered when predicting without events" do
      :rand.seed(:exsss, {6, 6, 6})
      df = holiday_frame(&(&1 == ~D[2023-03-15]))
      events_df = DataFrame.new(%{"event" => ["sale"], "ds" => [~D[2023-03-15]]})
      model = event_model(%{events: %{"sale" => %{steps_before: 0, steps_after: 0}}})
      fitted = Soothsayer.fit(model, df, events: events_df)

      predictions = Soothsayer.predict(fitted, Series.from_list([~D[2023-03-14], ~D[2023-03-15]]))
      [before, on_the_day] = Series.to_list(predictions["yhat"])
      assert on_the_day > before + 20
    end

    test "an event named like a holiday raises" do
      df = holiday_frame(fn _ -> false end)

      model =
        event_model(%{
          events: %{"Christmas Day" => %{steps_before: 0, steps_after: 0}},
          holidays: %{countries: [:us]}
        })

      assert_raise ArgumentError,
                   ~r/\["Christmas Day"\] are both configured events and country holidays/,
                   fn ->
                     Soothsayer.fit(model, df)
                   end
    end

    test "rejects bad event windows and recurrence" do
      assert_raise ArgumentError, ~r/events.sale.steps_before must be an integer >= 0/, fn ->
        Soothsayer.new(%{events: %{"sale" => %{steps_before: -1, steps_after: 0}}})
      end

      assert_raise ArgumentError, ~r/steps_before and steps_after now/, fn ->
        Soothsayer.new(%{events: %{"sale" => %{lower_window: -1, upper_window: 0}}})
      end

      assert_raise ArgumentError, ~r/events.sale.recurring must be :yearly/, fn ->
        Soothsayer.new(%{
          events: %{"sale" => %{steps_before: 0, steps_after: 0, recurring: :monthly}}
        })
      end
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
            "sale" => %{steps_before: 0, steps_after: 0}
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
            "promo" => %{steps_before: 1, steps_after: 1}
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
      assert DataFrame.n_rows(predictions) == 2
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
            "sale" => %{steps_before: 0, steps_after: 0}
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
            "promo" => %{steps_before: 1, steps_after: 1}
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
          events: %{"sale" => %{steps_before: 0, steps_after: 0}},
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

  describe "multiplicative events and regressors" do
    # y = trend * (1 + 0.3 on promo days) * (1 + 0.2 * x) + noise: both the
    # event and the regressor effects grow with the level of the series.
    defp scaled_frame(dates, start_date) do
      x = Enum.map(dates, fn date -> :math.sin(Date.diff(date, start_date) / 23) end)

      y =
        dates
        |> Enum.zip(x)
        |> Enum.map(fn {date, x_value} ->
          trend = 100 + 0.2 * Date.diff(date, start_date)
          promo = if date.day == 15, do: 0.3, else: 0.0
          trend * (1 + promo) * (1 + 0.2 * x_value) + :rand.normal(0, 2)
        end)

      DataFrame.new(%{"ds" => dates, "y" => y, "x" => x})
    end

    defp promo_frame(dates) do
      promo_dates = Enum.filter(dates, &(&1.day == 15))

      DataFrame.new(%{
        "event" => List.duplicate("promo", length(promo_dates)),
        "ds" => promo_dates
      })
    end

    defp scaled_model(event_mode, regressor_mode) do
      Soothsayer.new(%{
        trend: %{changepoints: 0},
        seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
        events: %{"promo" => %{mode: event_mode}},
        regressors: %{"x" => %{mode: regressor_mode}},
        epochs: 60,
        seed: 3
      })
    end

    test "multiplicative events and regressors follow the level where additive ones cannot" do
      :rand.seed(:exsss, {8, 8, 8})
      start_date = ~D[2018-01-01]
      training_dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      holdout_dates = Date.range(~D[2023-01-01], ~D[2023-12-31]) |> Enum.to_list()
      training = scaled_frame(training_dates, start_date)
      holdout = scaled_frame(holdout_dates, start_date)

      holdout_error = fn event_mode, regressor_mode ->
        fitted =
          Soothsayer.fit(scaled_model(event_mode, regressor_mode), training,
            events: promo_frame(training_dates)
          )

        predictions =
          Soothsayer.predict(fitted, holdout["ds"],
            events: promo_frame(holdout_dates),
            regressors: holdout
          )

        errors = Series.subtract(predictions["yhat"], holdout["y"]) |> Series.abs()
        promo_rows = Series.equal(Series.day_of_month(holdout["ds"]), 15)

        %{
          all: Series.mean(errors),
          promo_days: Series.mean(Series.mask(errors, promo_rows)),
          effects: Soothsayer.get_event_effects(fitted)
        }
      end

      additive = holdout_error.(:additive, :additive)
      multiplicative = holdout_error.(:multiplicative, :multiplicative)

      assert multiplicative.promo_days < additive.promo_days * 0.5
      assert multiplicative.all < additive.all * 0.7
      assert multiplicative.effects["promo_0"] > 0
    end

    test "components still sum to combined with both modes in play" do
      :rand.seed(:exsss, {8, 8, 8})
      start_date = ~D[2020-01-01]
      dates = Date.range(start_date, ~D[2021-12-31]) |> Enum.to_list()
      training = scaled_frame(dates, start_date)

      model =
        Soothsayer.new(%{
          trend: %{changepoints: 2},
          events: %{
            "promo" => %{mode: :multiplicative},
            "launch" => %{steps_before: 1, steps_after: 1}
          },
          regressors: %{"x" => %{mode: :multiplicative}},
          epochs: 3,
          seed: 3
        })

      events = promo_frame(dates)
      launch = DataFrame.new(%{"event" => ["launch"], "ds" => [~D[2020-06-01]]})
      fitted = Soothsayer.fit(model, training, events: DataFrame.concat_rows(events, launch))

      predictions =
        Soothsayer.predict(fitted, training["ds"], events: events, regressors: training)

      summed =
        ["trend", "yearly_seasonality", "weekly_seasonality", "events", "regressors"]
        |> Enum.map(&Series.to_tensor(predictions[&1]))
        |> Enum.reduce(&Nx.add/2)

      assert Nx.all_close(summed, Series.to_tensor(predictions["yhat"]), atol: 1.0e-2)
             |> Nx.to_number() == 1

      effects = Soothsayer.get_event_effects(fitted)

      assert Enum.sort(Map.keys(effects)) ==
               Enum.sort(["launch_-1", "launch_0", "launch_+1", "promo_0"])
    end

    test "regularization shrinks the coefficients it is put on" do
      :rand.seed(:exsss, {8, 8, 8})
      start_date = ~D[2020-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      training = scaled_frame(dates, start_date)
      noise_dates = Enum.filter(dates, &(&1.day == 3))

      events =
        DataFrame.concat_rows(
          promo_frame(dates),
          DataFrame.new(%{
            "event" => List.duplicate("noise", length(noise_dates)),
            "ds" => noise_dates
          })
        )

      fit_with = fn regularization ->
        model =
          Soothsayer.new(%{
            trend: %{changepoints: 0},
            seasonality: %{
              yearly: %{enabled: false},
              weekly: %{enabled: true},
              regularization: regularization
            },
            events: %{"promo" => %{}, "noise" => %{regularization: regularization}},
            epochs: 30,
            seed: 3
          })

        fitted = Soothsayer.fit(model, training, events: events)

        weekly =
          fitted.params.data["weekly_dense"]["kernel"] |> Nx.abs() |> Nx.sum() |> Nx.to_number()

        {Soothsayer.get_event_effects(fitted), weekly}
      end

      {plain_effects, plain_weekly} = fit_with.(nil)
      {shrunk_effects, shrunk_weekly} = fit_with.(1.0)

      # There is no weekly pattern and "noise" hits no real effect, so both shrink
      assert abs(shrunk_effects["noise_0"]) < abs(plain_effects["noise_0"]) * 0.5
      assert shrunk_weekly < plain_weekly * 0.5
      # The promo effect is real and unpenalized, so it stays
      assert shrunk_effects["promo_0"] > plain_effects["promo_0"] * 0.5
    end

    test "rejects an unknown mode" do
      assert_raise ArgumentError,
                   ~r/seasonality.regularization must be nil or a number >= 0/,
                   fn ->
                     Soothsayer.new(%{seasonality: %{regularization: -1}})
                   end

      assert_raise ArgumentError,
                   ~r/events.promo.regularization must be nil or a number >= 0/,
                   fn ->
                     Soothsayer.new(%{events: %{"promo" => %{regularization: "lots"}}})
                   end

      assert_raise ArgumentError, ~r/events.promo.mode must be one of/, fn ->
        Soothsayer.new(%{events: %{"promo" => %{mode: :scaled}}})
      end

      assert_raise ArgumentError, ~r/regressor "x" mode must be one of/, fn ->
        Soothsayer.new(%{regressors: %{"x" => %{mode: :scaled}}})
      end
    end
  end

  describe "discontinuous growth" do
    test "a level jump lands on the intercept of the segment it starts, not on the slopes" do
      :rand.seed(:exsss, {6, 6, 6})
      start_date = ~D[2020-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()

      # 10 changepoints over the first 80% of 1096 days sit every 79.7 days,
      # the sixth at day 478, where the level jumps by 40.
      y =
        Enum.map(dates, fn date ->
          days = Date.diff(date, start_date)
          jump = if days >= 478, do: 40.0, else: 0.0
          100 + 0.02 * days + jump + :rand.normal(0, 1)
        end)

      df = DataFrame.new(%{"ds" => dates, "y" => y})

      in_sample_rmse = fn growth ->
        model =
          Soothsayer.new(%{
            trend: %{changepoints: 10, growth: growth},
            seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
            epochs: 40,
            seed: 3
          })

        fitted = Soothsayer.fit(model, df)
        predictions = Soothsayer.predict(fitted, df["ds"])
        errors = Series.subtract(predictions["yhat"], df["y"])
        rmse = errors |> Series.pow(2) |> Series.mean() |> :math.sqrt()
        {rmse, fitted}
      end

      {linear_rmse, _} = in_sample_rmse.(:linear)
      {discontinuous_rmse, fitted} = in_sample_rmse.(:discontinuous)

      assert discontinuous_rmse < linear_rmse * 0.7

      # Segmentwise intercepts are one-hot, so every segment after the jump
      # carries the new level on its own and the ones before stay near zero.
      kernel = Soothsayer.Trend.get_weights(fitted).kernel |> Nx.flatten()
      intercepts = kernel[11..20] |> Nx.abs()
      before = intercepts[0..4] |> Nx.mean() |> Nx.to_number()
      after_jump = intercepts[5..9] |> Nx.mean() |> Nx.to_number()
      assert before < 0.25 * after_jump
      assert "trend" in DataFrame.names(Soothsayer.predict(fitted, df["ds"]))
    end

    test "rejects an unknown growth" do
      assert_raise ArgumentError,
                   ~r/trend.growth must be :linear, :discontinuous or :logistic/,
                   fn ->
                     Soothsayer.new(%{trend: %{growth: :off}})
                   end
    end
  end

  describe "logistic growth" do
    # An S curve approaching 1000: linear growth keeps climbing past it,
    # logistic growth levels off at the cap
    defp saturating_frame(dates, start_date, cap, seed) do
      :rand.seed(:exsss, {seed, seed, seed})

      y =
        Enum.map(dates, fn date ->
          cap / (1 + :math.exp(-0.008 * (Date.diff(date, start_date) - 500))) +
            :rand.normal(0, 10)
        end)

      DataFrame.new(%{"ds" => dates, "y" => y, "cap" => List.duplicate(cap, length(dates))})
    end

    defp saturating_model(trend, epochs \\ 60) do
      Soothsayer.new(%{
        trend: trend,
        seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
        epochs: epochs,
        seed: 1
      })
    end

    test "levels off at the cap where a linear trend keeps climbing" do
      start_date = ~D[2020-01-01]

      training =
        saturating_frame(
          Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list(),
          start_date,
          1000.0,
          1
        )

      holdout =
        saturating_frame(
          Date.range(~D[2023-01-01], ~D[2023-12-31]) |> Enum.to_list(),
          start_date,
          1000.0,
          2
        )

      errors =
        for trend <- [%{}, %{growth: :logistic}] do
          fitted = Soothsayer.fit(saturating_model(trend), training)
          predictions = Soothsayer.predict(fitted, holdout["ds"], regressors: holdout)

          {Series.subtract(predictions["yhat"], holdout["y"]) |> Series.abs() |> Series.mean(),
           Series.max(predictions["yhat"])}
        end

      [{linear_error, linear_max}, {logistic_error, logistic_max}] = errors
      assert logistic_error < linear_error * 0.5
      assert logistic_max < 1010.0
      assert linear_max > 1050.0
    end

    test "needs cap at fit and at predict, a floor when the data had one, and cap above floor" do
      dates = Date.range(~D[2022-01-01], ~D[2022-12-31]) |> Enum.to_list()
      frame = saturating_frame(dates, ~D[2022-01-01], 100.0, 3)

      assert_raise ArgumentError, ~r/needs a "cap" column/, fn ->
        Soothsayer.fit(
          saturating_model(%{growth: :logistic, changepoints: 0}),
          DataFrame.select(frame, ["ds", "y"])
        )
      end

      assert_raise ArgumentError, ~r/cap must be above its floor/, fn ->
        floored = DataFrame.put(frame, "floor", Series.from_list(List.duplicate(100.0, 365)))
        Soothsayer.fit(saturating_model(%{growth: :logistic, changepoints: 0}), floored)
      end

      floored = DataFrame.put(frame, "floor", Series.from_list(List.duplicate(-5.0, 365)))

      fitted = Soothsayer.fit(saturating_model(%{growth: :logistic, changepoints: 0}, 1), floored)
      assert fitted.config.trend.uses_floor

      assert Soothsayer.series_entry(fitted, nil).regressors["floor"][~N[2022-06-01 00:00:00]] ==
               -5.0

      future = DataFrame.new(%{"ds" => [~D[2023-01-01]], "cap" => [100.0]})

      assert_raise ArgumentError, ~r/capacity columns \["cap", "floor"\]/, fn ->
        Soothsayer.predict(fitted, future["ds"])
      end

      assert_raise ArgumentError, ~r/Regressor column "floor" not found/, fn ->
        Soothsayer.predict(fitted, future["ds"], regressors: future)
      end

      with_floor = DataFrame.put(future, "floor", Series.from_list([-5.0]))
      predictions = Soothsayer.predict(fitted, future["ds"], regressors: with_floor)
      assert Series.first(predictions["yhat"]) |> is_float()
    end
  end

  describe "changepoints at known dates" do
    # A slope that bends once, at a date the grid doesn't land on
    defp bend_frame(dates, start_date, bend_date, seed) do
      :rand.seed(:exsss, {seed, seed, seed})

      y =
        Enum.map(dates, fn date ->
          days = Date.diff(date, start_date)
          after_bend = max(Date.diff(date, bend_date), 0)
          100 + 0.01 * days + 0.08 * after_bend + :rand.normal(0, 0.5)
        end)

      DataFrame.new(%{"ds" => dates, "y" => y})
    end

    test "naming the bend beats the grid, and the dates replace it" do
      start_date = ~D[2019-01-01]
      bend_date = ~D[2022-06-15]

      training =
        bend_frame(
          Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list(),
          start_date,
          bend_date,
          1
        )

      holdout =
        bend_frame(
          Date.range(~D[2023-01-01], ~D[2023-03-31]) |> Enum.to_list(),
          start_date,
          bend_date,
          2
        )

      config = %{
        seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
        epochs: 30,
        seed: 3
      }

      errors =
        for changepoints <- [10, [bend_date]] do
          model = Soothsayer.new(Map.put(config, :trend, %{changepoints: changepoints}))
          fitted = Soothsayer.fit(model, training)
          predictions = Soothsayer.predict(fitted, holdout["ds"])

          {fitted,
           Series.subtract(predictions["yhat"], holdout["y"]) |> Series.abs() |> Series.mean()}
        end

      [{_grid, grid_error}, {named, named_error}] = errors
      assert named_error < grid_error * 0.5
      assert named.config.changepoint_positions == [Date.diff(bend_date, start_date) * 1.0]
      assert Nx.shape(Soothsayer.Trend.get_weights(named).kernel) == {2, 1}
    end

    test "dates must be sorted, unique, of the right type and inside the data" do
      assert_raise ArgumentError, ~r/sorted and unique/, fn ->
        Soothsayer.new(%{trend: %{changepoints: [~D[2022-06-01], ~D[2021-01-01]]}})
      end

      assert_raise ArgumentError, ~r/must be Date or NaiveDateTime/, fn ->
        Soothsayer.new(%{trend: %{changepoints: ["2022-06-01"]}})
      end

      assert_raise ArgumentError, ~r/count or a non-empty list/, fn ->
        Soothsayer.new(%{trend: %{changepoints: []}})
      end

      dates = Date.range(~D[2022-01-01], ~D[2022-12-31]) |> Enum.to_list()
      frame = DataFrame.new(%{"ds" => dates, "y" => Enum.map(dates, &Date.day_of_year/1)})

      assert_raise ArgumentError, ~r/outside the training data/, fn ->
        Soothsayer.fit(
          Soothsayer.new(%{trend: %{changepoints: [~D[2023-06-01]]}, epochs: 1}),
          frame
        )
      end
    end
  end

  describe "custom and conditional seasonalities" do
    defp cycle_frame(dates, start_date, period, summer_only?) do
      y =
        Enum.map(dates, fn date ->
          days = Date.diff(date, start_date)
          summer = if date.month in 6..8, do: 1.0, else: 0.0
          weekly = 8 * :math.sin(2 * :math.pi() * Date.day_of_week(date) / 7)
          cycle = 10 * :math.sin(2 * :math.pi() * days / period)
          seasonal = if summer_only?, do: weekly * summer, else: cycle
          100 + 0.02 * days + seasonal + :rand.normal(0, 1)
        end)

      summer = Enum.map(dates, &(&1.month in 6..8))
      DataFrame.new(%{"ds" => dates, "y" => y, "summer" => summer})
    end

    defp cycle_model(seasonality) do
      Soothsayer.new(%{
        trend: %{changepoints: 0},
        seasonality:
          Map.merge(%{yearly: %{enabled: false}, weekly: %{enabled: false}}, seasonality),
        epochs: 40,
        seed: 3
      })
    end

    defp holdout_mae(model, training, holdout) do
      fitted = Soothsayer.fit(model, training)
      predictions = Soothsayer.predict(fitted, holdout["ds"], regressors: holdout)

      {Series.subtract(predictions["yhat"], holdout["y"]) |> Series.abs() |> Series.mean(),
       predictions}
    end

    test "a custom period picks up a cycle the built-in periods can't" do
      :rand.seed(:exsss, {2, 2, 2})
      start_date = ~D[2020-01-01]
      training_dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      holdout_dates = Date.range(~D[2023-01-01], ~D[2023-06-30]) |> Enum.to_list()
      training = cycle_frame(training_dates, start_date, 30.5, false)
      holdout = cycle_frame(holdout_dates, start_date, 30.5, false)

      {without_error, _} = holdout_mae(cycle_model(%{}), training, holdout)

      {with_error, predictions} =
        holdout_mae(
          cycle_model(%{custom: %{"monthly" => %{period: 30.5, fourier_terms: 3}}}),
          training,
          holdout
        )

      assert with_error < without_error * 0.4
      assert "monthly_seasonality" in DataFrame.names(predictions)
      refute "yearly_seasonality" in DataFrame.names(predictions)
    end

    test "a conditional weekly pattern beats one that has to apply all year" do
      :rand.seed(:exsss, {2, 2, 2})
      start_date = ~D[2020-01-01]
      training_dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      holdout_dates = Date.range(~D[2023-01-01], ~D[2023-12-31]) |> Enum.to_list()
      training = cycle_frame(training_dates, start_date, 7, true)
      holdout = cycle_frame(holdout_dates, start_date, 7, true)

      {plain_error, _} = holdout_mae(cycle_model(%{weekly: %{enabled: true}}), training, holdout)

      {conditional_error, predictions} =
        holdout_mae(
          cycle_model(%{weekly: %{enabled: true, condition: "summer"}}),
          training,
          holdout
        )

      assert conditional_error < plain_error * 0.7

      winter =
        Series.mask(predictions["weekly_seasonality"], Series.equal(holdout["summer"], false))

      # Outside the condition the features are zero, which z-scoring puts a hair off 0
      assert Series.max(Series.abs(winter)) < 0.1

      fitted =
        Soothsayer.fit(cycle_model(%{weekly: %{enabled: true, condition: "summer"}}), training)

      assert_raise ArgumentError, ~r/seasonality conditions \["summer"\]/, fn ->
        Soothsayer.predict(fitted, holdout["ds"])
      end
    end

    test "conditions cover the rows that missing data handling adds" do
      :rand.seed(:exsss, {2, 2, 2})
      start_date = ~D[2020-01-01]
      dates = Date.range(start_date, ~D[2021-12-31]) |> Enum.to_list()
      frame = cycle_frame(dates, start_date, 7, true)
      gapped = DataFrame.filter_with(frame, &Series.not_equal(&1["ds"], ~D[2020-07-15]))

      model =
        Soothsayer.new(%{
          trend: %{changepoints: 0},
          seasonality: %{
            yearly: %{enabled: false},
            weekly: %{enabled: true, condition: "summer"}
          },
          ar: %{enabled: true, lags: 2},
          epochs: 1
        })

      fitted = Soothsayer.fit(model, gapped)

      # the regridded row got an imputed condition value along with its y
      assert length(Soothsayer.series_entry(fitted, nil).timestamps) ==
               DataFrame.n_rows(gapped) + 1

      assert Soothsayer.series_entry(fitted, nil).regressors["summer"][~N[2020-07-15 00:00:00]] ==
               1.0
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

        predictions["yhat"]
        |> Series.to_tensor()
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

      # The trend scale must not turn the disabled periods into full tensors,
      # or predict reports them as columns of zeros.
      assert Nx.size(components.weekly_seasonality) == 1
      assert Nx.size(components.daily_seasonality) == 1

      predictions = Soothsayer.predict(fitted_model, Series.from_list(future_dates))
      assert DataFrame.names(predictions) == ["ds", "yhat", "trend", "yearly_seasonality"]
    end

    test "only the trend has an intercept, so the seasonality has no level of its own" do
      :rand.seed(:exsss, {5, 6, 7})
      start_date = ~D[2020-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      df = DataFrame.new(%{"ds" => dates, "y" => multiplicative_series(dates, start_date)})
      fitted_model = Soothsayer.fit(Soothsayer.new(%{epochs: 5}), df)

      assert Map.has_key?(fitted_model.params.data["trend_dense"], "bias")
      refute Map.has_key?(fitted_model.params.data["yearly_dense"], "bias")
      refute Map.has_key?(fitted_model.params.data["weekly_dense"], "bias")
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
      assert DataFrame.n_rows(Soothsayer.predict(fitted, Series.from_list([~D[2023-01-11]]))) == 1

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
          events: %{"sale" => %{steps_before: 0, steps_after: 0}},
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

    test "rejects bad seasonality enabled values" do
      assert_raise ArgumentError,
                   ~r/seasonality.daily.enabled must be true, false or :auto/,
                   fn ->
                     Soothsayer.new(%{seasonality: %{daily: %{enabled: :always}}})
                   end
    end
  end

  describe "missing data" do
    defp gapped_frame(rows, missing_indices) do
      dates = Enum.map(0..(rows - 1), &Date.add(~D[2023-01-01], &1))

      y =
        Enum.map(0..(rows - 1), fn index ->
          if index in missing_indices, do: nil, else: 10.0 + index * 0.5
        end)

      {dates, DataFrame.new(%{"ds" => dates, "y" => y})}
    end

    defp ar_model(overrides \\ %{}) do
      Soothsayer.new(
        Map.merge(
          %{
            seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
            ar: %{enabled: true, lags: 3},
            epochs: 1,
            learning_rate: 0.01
          },
          overrides
        )
      )
    end

    test "without auto-regression the rows with a missing y are dropped" do
      {dates, data} = gapped_frame(40, [5, 6])
      fitted = Soothsayer.fit(Soothsayer.new(%{epochs: 1, learning_rate: 0.01}), data)

      assert length(Soothsayer.series_entry(fitted, nil).timestamps) == 38

      refute Enum.at(dates, 5) in Enum.map(
               Soothsayer.series_entry(fitted, nil).timestamps,
               &NaiveDateTime.to_date/1
             )

      predictions = Soothsayer.predict(fitted, Series.from_list(Enum.take(dates, 3)))
      assert DataFrame.n_rows(predictions) == 3
    end

    test "with auto-regression a short gap is imputed linearly" do
      {dates, data} = gapped_frame(40, [10, 11])
      fitted = Soothsayer.fit(ar_model(), data)

      %{mean: mean, std: std} = fitted.config.normalization.y
      mean = mean |> Nx.squeeze() |> Nx.to_number()
      std = std |> Nx.squeeze() |> Nx.to_number()
      gap_timestamp = dates |> Enum.at(10) |> NaiveDateTime.new!(~T[00:00:00])

      assert length(Soothsayer.series_entry(fitted, nil).timestamps) == 40

      assert_in_delta Soothsayer.series_entry(fitted, nil).known_values[gap_timestamp],
                      (15.0 - mean) / std,
                      1.0e-4
    end

    test "with auto-regression a missing row is inserted on the grid" do
      {dates, data} = gapped_frame(40, [])
      without_row = DataFrame.filter_with(data, &Series.not_equal(&1["ds"], Enum.at(dates, 20)))
      assert DataFrame.n_rows(without_row) == 39

      fitted = Soothsayer.fit(ar_model(), without_row)
      assert length(Soothsayer.series_entry(fitted, nil).timestamps) == 40
      assert Soothsayer.series_entry(fitted, nil).known_values |> Map.keys() |> length() == 40
    end

    test "with auto-regression an off-grid row raises" do
      # daily rows on a two day grid: the second row is already off it
      {_dates, data} = gapped_frame(41, [])

      assert_raise ArgumentError, ~r/2023-01-02 is not on the 2 day grid/, fn ->
        Soothsayer.fit(ar_model(%{frequency: {2, :day}}), data)
      end
    end

    test "a gap too long to impute raises unless samples may be dropped" do
      {dates, data} = gapped_frame(120, Enum.to_list(40..79))

      assert_raise ArgumentError,
                   ~r/training samples touch missing values that couldn't be imputed \(gaps longer than 30 steps\)/,
                   fn -> Soothsayer.fit(ar_model(), data) end

      fitted = Soothsayer.fit(ar_model(%{missing: %{drop_samples: true}}), data)
      known = Soothsayer.series_entry(fitted, nil).known_values

      # 10 imputed from each side, the 20 in the middle stay unknown
      assert map_size(known) == 100
      refute Map.has_key?(known, NaiveDateTime.new!(Enum.at(dates, 60), ~T[00:00:00]))
      assert Map.has_key?(known, NaiveDateTime.new!(Enum.at(dates, 45), ~T[00:00:00]))
    end

    test "imputation can be turned off" do
      {_dates, data} = gapped_frame(40, [10])

      assert_raise ArgumentError, ~r/training samples touch missing values/, fn ->
        Soothsayer.fit(ar_model(%{missing: %{impute: false}}), data)
      end
    end

    test "rejects bad missing options" do
      assert_raise ArgumentError, ~r/missing.impute_linear must be a non-negative integer/, fn ->
        Soothsayer.new(%{missing: %{impute_linear: -1}})
      end

      assert_raise ArgumentError, ~r/missing.drop_samples must be true or false/, fn ->
        Soothsayer.new(%{missing: %{drop_samples: :maybe}})
      end
    end
  end

  describe "predict dataframe" do
    defp daily_frame do
      dates = Date.range(~D[2021-01-01], ~D[2022-12-31]) |> Enum.to_list()

      y =
        Enum.map(dates, fn date ->
          day = Date.diff(date, ~D[2021-01-01])
          100 + 0.1 * day + 10 * :math.sin(2 * :math.pi() * day / 365.25) + rem(day, 7)
        end)

      {dates, DataFrame.new(%{"ds" => dates, "y" => y})}
    end

    test "has ds, yhat, quantile and enabled component columns in order" do
      {dates, df} = daily_frame()

      model =
        Soothsayer.new(%{
          seasonality: %{weekly: %{enabled: false}},
          quantiles: [0.975, 0.1],
          epochs: 2,
          learning_rate: 0.01,
          seed: 1
        })

      future = Series.from_list(Enum.map(1..5, &Date.add(List.last(dates), &1)))
      predictions = Soothsayer.fit(model, df) |> Soothsayer.predict(future)

      assert DataFrame.names(predictions) ==
               ["ds", "yhat", "yhat_10", "yhat_97.5", "trend", "yearly_seasonality"]

      assert DataFrame.n_rows(predictions) == 5
      assert Series.dtype(predictions["ds"]) == :date
      assert Series.to_list(predictions["ds"]) == Series.to_list(future)

      assert Series.to_list(predictions["yhat_10"])
             |> Enum.zip(Series.to_list(predictions["yhat"]))
             |> Enum.all?(fn {low, median} -> low <= median end)
    end

    test "keeps the timestamp dtype of the input" do
      hours = Enum.map(0..99, &NaiveDateTime.add(~N[2023-01-01 00:00:00], &1, :hour))
      y = Enum.map(0..99, &(10 + :math.sin(&1 / 24 * 2 * :math.pi())))
      df = DataFrame.new(%{"ds" => hours, "y" => y})

      model =
        Soothsayer.new(%{
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          epochs: 1,
          learning_rate: 0.01
        })

      next = Series.from_list([~N[2023-01-05 04:00:00], ~N[2023-01-05 05:00:00]])
      predictions = Soothsayer.fit(model, df) |> Soothsayer.predict(next)

      assert {:naive_datetime, _precision} = Series.dtype(predictions["ds"])
      assert DataFrame.names(predictions) == ["ds", "yhat", "trend", "daily_seasonality"]
    end

    test "component columns add up to yhat, with the trend on or off" do
      {dates, df} = daily_frame()
      future = Series.from_list(Enum.map(1..10, &Date.add(List.last(dates), &1)))

      for trend_enabled <- [true, false] do
        model =
          Soothsayer.new(%{
            trend: %{enabled: trend_enabled},
            epochs: 2,
            learning_rate: 0.01,
            seed: 1
          })

        predictions = Soothsayer.fit(model, df) |> Soothsayer.predict(future)

        assert DataFrame.names(predictions) == [
                 "ds",
                 "yhat",
                 "trend",
                 "yearly_seasonality",
                 "weekly_seasonality"
               ]

        summed =
          ["trend", "yearly_seasonality", "weekly_seasonality"]
          |> Enum.map(&predictions[&1])
          |> Enum.reduce(&Series.add/2)

        difference =
          summed |> Series.subtract(predictions["yhat"]) |> Series.abs() |> Series.max()

        assert difference < 1.0e-3

        unless trend_enabled do
          # a flat line at the training mean
          assert Series.n_distinct(predictions["trend"]) == 1
        end
      end
    end
  end

  describe "recency weighting" do
    # A slope that steepens in the last third: a single line fitted with
    # recent rows weighted follows the new slope, a flat weight averages
    # the two.
    defp bent_series(dates, start_date, bend_date) do
      Enum.map(dates, fn date ->
        days = Date.diff(date, start_date)
        bent_days = max(Date.diff(date, bend_date), 0)
        100 + 0.02 * days + 0.06 * bent_days + :rand.normal(0, 0.5)
      end)
    end

    test "weighting recent rows follows a slope that changed late in the training data" do
      :rand.seed(:exsss, {9, 9, 9})
      start_date = ~D[2021-01-01]
      bend_date = ~D[2022-04-01]
      training_dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      holdout_dates = Date.range(~D[2023-01-01], ~D[2023-01-31]) |> Enum.to_list()

      training =
        DataFrame.new(%{
          "ds" => training_dates,
          "y" => bent_series(training_dates, start_date, bend_date)
        })

      holdout =
        DataFrame.new(%{
          "ds" => holdout_dates,
          "y" => bent_series(holdout_dates, start_date, bend_date)
        })

      config = %{
        trend: %{changepoints: 0},
        seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
        epochs: 40,
        seed: 4
      }

      errors =
        for recency <- [%{enabled: false}, %{weight: 10, start: 0.7}] do
          model = Soothsayer.new(Map.put(config, :recency, recency))
          fitted = Soothsayer.fit(model, training)
          predictions = Soothsayer.predict(fitted, holdout["ds"])
          Series.subtract(predictions["yhat"], holdout["y"]) |> Series.abs() |> Series.mean()
        end

      [flat_error, weighted_error] = errors
      assert weighted_error < flat_error * 0.6
    end

    test "rejects a weight below 1, a start outside [0, 1) or a non-boolean enabled" do
      assert_raise ArgumentError, ~r/recency must be/, fn ->
        Soothsayer.new(%{recency: %{weight: 0.5}})
      end

      assert_raise ArgumentError, ~r/recency must be/, fn ->
        Soothsayer.new(%{recency: %{start: 1.0}})
      end

      assert_raise ArgumentError, ~r/recency must be/, fn ->
        Soothsayer.new(%{recency: %{enabled: nil}})
      end

      assert Soothsayer.new(%{recency: %{enabled: false}}).config.recency ==
               %{enabled: false, weight: 2, start: 0.0}
    end
  end

  describe "conformal prediction" do
    defp noisy_frame(dates, start_date) do
      y =
        Enum.map(dates, fn date ->
          days = Date.diff(date, start_date)
          50 + 0.03 * days + 5 * :math.sin(2 * :math.pi() * days / 365.25) + :rand.normal(0, 3)
        end)

      DataFrame.new(%{"ds" => dates, "y" => y})
    end

    defp coverage(predictions, holdout) do
      inside =
        Series.and(
          Series.greater_equal(holdout["y"], predictions["yhat_lower"]),
          Series.less_equal(holdout["y"], predictions["yhat_upper"])
        )

      inside |> Series.cast({:s, 8}) |> Series.mean()
    end

    test "calibrated intervals cover about 1 - alpha of unseen points, with both methods" do
      :rand.seed(:exsss, {11, 11, 11})
      start_date = ~D[2019-01-01]
      training = noisy_frame(Date.range(start_date, ~D[2021-12-31]) |> Enum.to_list(), start_date)

      calibration =
        noisy_frame(Date.range(~D[2022-01-01], ~D[2022-06-30]) |> Enum.to_list(), start_date)

      holdout =
        noisy_frame(Date.range(~D[2022-07-01], ~D[2022-12-31]) |> Enum.to_list(), start_date)

      fitted =
        Soothsayer.fit(
          Soothsayer.new(%{
            trend: %{changepoints: 0},
            seasonality: %{weekly: %{enabled: false}},
            quantiles: [0.1, 0.9],
            epochs: 40,
            seed: 5
          }),
          training
        )

      naive = Soothsayer.calibrate(fitted, calibration, alpha: 0.1)
      naive_predictions = Soothsayer.predict(naive, holdout["ds"])

      assert naive.config.calibration.method == :naive
      assert map_size(naive.config.calibration.q_hat) == 1

      assert DataFrame.names(naive_predictions) |> Enum.take(5) ==
               ["ds", "yhat", "yhat_10", "yhat_90", "yhat_lower"]

      naive_coverage = coverage(naive_predictions, holdout)
      assert naive_coverage > 0.8 and naive_coverage < 0.98

      cqr = Soothsayer.calibrate(fitted, calibration, alpha: 0.1, method: :cqr)
      cqr_predictions = Soothsayer.predict(cqr, holdout["ds"])
      cqr_coverage = coverage(cqr_predictions, holdout)
      assert cqr_coverage > 0.8 and cqr_coverage < 0.98

      # cqr keeps the quantile columns as they were and moves the band out
      # from them by q_hat on each side
      assert Series.to_list(cqr_predictions["yhat_10"]) ==
               Series.to_list(naive_predictions["yhat_10"])

      q = cqr.config.calibration.q_hat[1]

      assert_in_delta Series.first(
                        Series.subtract(cqr_predictions["yhat_10"], cqr_predictions["yhat_lower"])
                      ),
                      q,
                      1.0e-4

      # the backtest reports the calibrated interval's coverage and width
      result =
        Soothsayer.backtest(cqr, DataFrame.concat_rows(training, holdout),
          validation_fraction: 0.05
        )

      assert result.metrics.coverage > 0.5
      assert result.metrics.mean_interval_width > 0
      assert "yhat_lower" in DataFrame.names(result.predictions)
    end

    test "with auto-regression every forecast step gets its own q_hat" do
      :rand.seed(:exsss, {12, 12, 12})
      start_date = ~D[2021-01-01]
      training = noisy_frame(Date.range(start_date, ~D[2022-06-30]) |> Enum.to_list(), start_date)

      calibration =
        noisy_frame(Date.range(~D[2022-07-01], ~D[2022-09-30]) |> Enum.to_list(), start_date)

      fitted =
        Soothsayer.fit(
          Soothsayer.new(%{
            trend: %{changepoints: 0},
            seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
            ar: %{enabled: true, lags: 7, forecast_steps: 3},
            epochs: 5,
            seed: 6
          }),
          training
        )

      calibrated = Soothsayer.calibrate(fitted, calibration, alpha: 0.2)

      assert Map.keys(calibrated.config.calibration.q_hat) |> Enum.sort() == [1, 2, 3]
      assert Enum.all?(Map.values(calibrated.config.calibration.q_hat), &(&1 > 0))

      # ten days out: the first three rows are calibrated steps, the rest use step 3
      future = Series.from_list(Date.range(~D[2022-10-01], ~D[2022-10-10]) |> Enum.to_list())
      predictions = Soothsayer.predict(calibrated, future, history: calibration)

      widths =
        Series.subtract(predictions["yhat_upper"], predictions["yhat_lower"]) |> Series.to_list()

      assert Enum.drop(widths, 2) |> Enum.uniq_by(&Float.round(&1, 4)) |> length() == 1
      assert_in_delta hd(widths), 2 * calibrated.config.calibration.q_hat[1], 1.0e-4
    end
  end

  describe "several series" do
    # Two series with the same seasonal amplitude and noise, shifted in
    # level, so one seasonality kernel fits both once each is z-scored.
    defp panel_frame(ids_and_levels, dates, start_date, seed) do
      :rand.seed(:exsss, {seed, seed, seed})

      ids_and_levels
      |> Enum.map(fn {id, level} ->
        y =
          Enum.map(dates, fn date ->
            days = Date.diff(date, start_date)

            level + 0.02 * days + 5 * :math.sin(2 * :math.pi() * days / 365.25) +
              :rand.normal(0, 1)
          end)

        DataFrame.new(%{"ds" => dates, "y" => y, "id" => List.duplicate(id, length(dates))})
      end)
      |> DataFrame.concat_rows()
    end

    defp rows_of(frame, id), do: DataFrame.filter_with(frame, &Series.equal(&1["id"], id))

    defp mae(predictions, actual) do
      Series.subtract(predictions["yhat"], actual["y"]) |> Series.abs() |> Series.mean()
    end

    test "one model over two series forecasts each about as well as its own fit" do
      start_date = ~D[2020-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      future_dates = Date.range(~D[2023-01-01], ~D[2023-03-31]) |> Enum.to_list()
      levels = [{"a", 100}, {"b", 300}]
      training = panel_frame(levels, dates, start_date, 1)
      holdout = panel_frame(levels, future_dates, start_date, 2)
      config = %{epochs: 20, seed: 1, trend: %{changepoints: 0}}

      fitted = Soothsayer.fit(Soothsayer.new(Map.put(config, :series, %{column: "id"})), training)
      assert fitted.config.series.ids == ["a", "b"]

      # rows in a shuffled order come back in that order, with the id column
      shuffled = DataFrame.slice(holdout, Enum.shuffle(0..(DataFrame.n_rows(holdout) - 1)))
      predictions = Soothsayer.predict(fitted, DataFrame.select(shuffled, ["ds", "id"]))

      assert DataFrame.names(predictions) |> Enum.take(3) == ["ds", "id", "yhat"]
      assert Series.to_list(predictions["ds"]) == Series.to_list(shuffled["ds"])
      assert Series.to_list(predictions["id"]) == Series.to_list(shuffled["id"])

      for {id, level} <- levels do
        single =
          Soothsayer.fit(
            Soothsayer.new(config),
            DataFrame.select(rows_of(training, id), ["ds", "y"])
          )

        single_error =
          mae(Soothsayer.predict(single, rows_of(holdout, id)["ds"]), rows_of(holdout, id))

        panel_error = mae(rows_of(predictions, id), rows_of(shuffled, id))
        assert panel_error < single_error * 1.3
        assert_in_delta Series.mean(rows_of(predictions, id)["trend"]), level + 0.02 * 1140, 15
      end

      # the components still add up per row
      components = Soothsayer.predict_components(fitted, DataFrame.select(holdout, ["ds", "id"]))
      parts = for key <- [:trend, :yearly_seasonality, :weekly_seasonality], do: components[key]
      summed = Enum.reduce(parts, &Nx.add/2)
      assert Nx.all_close(summed, components.combined, atol: 1.0e-2) |> Nx.to_number() == 1
    end

    test "predict wants a frame with the id column and known ids" do
      start_date = ~D[2022-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      training = panel_frame([{"a", 10}, {"b", 20}], dates, start_date, 3)

      fitted =
        Soothsayer.fit(
          Soothsayer.new(%{series: %{column: "id"}, epochs: 1, trend: %{changepoints: 0}}),
          training
        )

      future = Series.from_list([~D[2023-01-01]])

      assert_raise ArgumentError, ~r/fitted on several series/, fn ->
        Soothsayer.predict(fitted, future)
      end

      assert_raise ArgumentError, ~r/Unknown series \["c"\]/, fn ->
        Soothsayer.predict(fitted, DataFrame.new(%{"ds" => [~D[2023-01-01]], "id" => ["c"]}))
      end

      single =
        Soothsayer.fit(
          Soothsayer.new(%{epochs: 1}),
          DataFrame.select(rows_of(training, "a"), ["ds", "y"])
        )

      assert_raise ArgumentError, ~r/fitted on a single series/, fn ->
        Soothsayer.predict(single, DataFrame.new(%{"ds" => [~D[2023-01-01]], "id" => ["a"]}))
      end
    end

    test "an events frame with the id column hits one series, without it every series" do
      start_date = ~D[2022-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      training = panel_frame([{"a", 10}, {"b", 20}], dates, start_date, 5)

      model =
        Soothsayer.new(%{
          series: %{column: "id"},
          events: %{"promo" => %{steps_before: 0, steps_after: 0}},
          trend: %{changepoints: 0},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          epochs: 1
        })

      future = DataFrame.new(%{"ds" => [~D[2023-03-01], ~D[2023-03-01]], "id" => ["a", "b"]})

      per_series =
        DataFrame.new(%{
          "event" => ["promo", "promo"],
          "ds" => [~D[2022-03-01], ~D[2023-03-01]],
          "id" => ["a", "a"]
        })

      fitted = Soothsayer.fit(model, training, events: per_series)

      assert Soothsayer.series_entry(fitted, "a").event_dates ==
               %{"promo" => [~N[2022-03-01 00:00:00], ~N[2023-03-01 00:00:00]]}

      assert Soothsayer.series_entry(fitted, "b").event_dates == %{}

      # A zero indicator is z-scored a hair off zero, so "no event" is the
      # value of a date with no events at all rather than 0.
      quiet = DataFrame.new(%{"ds" => [~D[2023-05-01], ~D[2023-05-01]], "id" => ["a", "b"]})
      [_, no_event] = Soothsayer.predict_components(fitted, quiet).events |> Nx.to_flat_list()

      # at predict the frame again names the series, only "a" gets the effect
      events = Soothsayer.predict_components(fitted, future, events: per_series).events
      [effect_a, effect_b] = Nx.to_flat_list(events)
      assert effect_a != no_event
      assert_in_delta effect_b, no_event, 1.0e-6

      # the same frame without ids is for everyone: now "b" gets it too (in
      # its own units, so the two effects differ by the series' scales)
      shared = DataFrame.select(per_series, ["event", "ds"])
      events = Soothsayer.predict_components(fitted, future, events: shared).events
      [shared_a, shared_b] = Nx.to_flat_list(events)
      assert_in_delta shared_a, effect_a, 1.0e-6
      assert abs(shared_b - no_event) > 1.0

      # the backtest splits the events frame the same way
      result = Soothsayer.backtest(model, training, validation_fraction: 0.02, events: per_series)
      assert "id" in DataFrame.names(result.predictions)
    end

    test "an unknown id is forecast with the shared components when asked for" do
      start_date = ~D[2020-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      levels = [{"a", 100}, {"b", 300}]
      training = panel_frame(levels, dates, start_date, 6)

      future =
        DataFrame.new(%{
          "ds" => [~D[2023-01-15], ~D[2023-01-15], ~D[2023-01-15]],
          "id" => ["a", "b", "c"]
        })

      config = %{epochs: 10, seed: 1, trend: %{changepoints: 0}}

      strict = Soothsayer.fit(Soothsayer.new(Map.put(config, :series, %{column: "id"})), training)

      assert_raise ArgumentError, ~r/Unknown series \["c"\].*series.unknown: :global/, fn ->
        Soothsayer.predict(strict, future)
      end

      lenient =
        Soothsayer.fit(
          Soothsayer.new(Map.put(config, :series, %{column: "id", unknown: :global})),
          training
        )

      predictions = Soothsayer.predict(lenient, future)
      [yhat_a, _yhat_b, yhat_c] = Series.to_list(predictions["yhat"])
      assert Series.to_list(predictions["id"]) == ["a", "b", "c"]

      # the same shared forecast in normalized space, put on the global scale
      normalized = fn yhat, %{mean: mean, std: std} ->
        (yhat - Nx.to_number(Nx.squeeze(mean))) / Nx.to_number(Nx.squeeze(std))
      end

      global = lenient.config.training_data.global_normalization

      assert_in_delta normalized.(yhat_c, global),
                      normalized.(yhat_a, Soothsayer.series_entry(lenient, "a").normalization),
                      1.0e-3

      # with auto-regression an unknown series needs history for its lags
      ar_model =
        Soothsayer.new(
          Map.merge(config, %{
            series: %{column: "id", unknown: :global},
            ar: %{enabled: true, lags: 3, forecast_steps: 1},
            epochs: 2
          })
        )

      ar_fitted = Soothsayer.fit(ar_model, training)
      only_c = DataFrame.new(%{"ds" => [~D[2023-01-01]], "id" => ["c"]})

      assert_raise ArgumentError,
                   ~r/was not seen at fit and the model uses auto-regression/,
                   fn ->
                     Soothsayer.predict(ar_fitted, only_c)
                   end

      history =
        DataFrame.new(%{
          "ds" => Date.range(~D[2022-12-29], ~D[2022-12-31]) |> Enum.to_list(),
          "y" => [200.0, 201.0, 202.0],
          "id" => ["c", "c", "c"]
        })

      with_history = Soothsayer.predict(ar_fitted, only_c, history: history)
      assert Series.first(with_history["yhat"]) |> is_float()
      assert abs(Series.first(with_history["yhat"]) - 200) < 40
    end

    test "series with different frequencies are refused" do
      daily =
        DataFrame.new(%{
          "ds" => Date.range(~D[2022-01-01], ~D[2022-01-31]) |> Enum.to_list(),
          "y" => Enum.to_list(1..31),
          "id" => List.duplicate("a", 31)
        })

      weekly =
        DataFrame.new(%{
          "ds" => Date.range(~D[2022-01-01], ~D[2022-03-31], 7) |> Enum.to_list(),
          "y" => Enum.to_list(1..13),
          "id" => List.duplicate("b", 13)
        })

      assert_raise ArgumentError, ~r/same frequency/, fn ->
        Soothsayer.fit(
          Soothsayer.new(%{series: %{column: "id"}, epochs: 1}),
          DataFrame.concat_rows([daily, weekly])
        )
      end
    end

    test "auto-regression seeds each series from its own history and the backtest walks each series" do
      start_date = ~D[2021-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      levels = [{"a", 100}, {"b", 300}]
      training = panel_frame(levels, dates, start_date, 4)

      model =
        Soothsayer.new(%{
          series: %{column: "id"},
          ar: %{enabled: true, lags: 7, forecast_steps: 2},
          trend: %{changepoints: 0},
          epochs: 5,
          seed: 2
        })

      result = Soothsayer.backtest(model, training, validation_fraction: 0.02)

      assert DataFrame.names(result.predictions) |> Enum.take(3) == ["id", "origin", "ds"]

      assert result.predictions["id"] |> Series.distinct() |> Series.to_list() |> Enum.sort() == [
               "a",
               "b"
             ]

      assert result.metrics.mean_absolute_error < 10

      # history for one series only moves that series' forecast
      fitted = Soothsayer.fit(model, training)
      future = DataFrame.new(%{"ds" => [~D[2023-01-02], ~D[2023-01-02]], "id" => ["a", "b"]})
      plain = Soothsayer.predict(fitted, future)
      history = DataFrame.new(%{"ds" => [~D[2023-01-01]], "y" => [500.0], "id" => ["a"]})
      with_history = Soothsayer.predict(fitted, future, history: history)

      assert abs(Series.first(with_history["yhat"]) - Series.first(plain["yhat"])) > 5
      assert Series.last(with_history["yhat"]) == Series.last(plain["yhat"])

      assert_raise ArgumentError, ~r/history frame needs the "id" column/, fn ->
        Soothsayer.predict(fitted, future, history: DataFrame.select(history, ["ds", "y"]))
      end
    end
  end

  describe "local trend and seasonality" do
    # Two series with their own slope and opposite yearly phase: shared
    # kernels can only fit the average of the two.
    defp diverging_panel(dates, start_date, seed) do
      :rand.seed(:exsss, {seed, seed, seed})

      [{"a", 100, 0.05, 0.0}, {"b", 300, -0.02, :math.pi()}]
      |> Enum.map(fn {id, level, slope, phase} ->
        y =
          Enum.map(dates, fn date ->
            days = Date.diff(date, start_date)

            level + slope * days + 5 * :math.sin(2 * :math.pi() * days / 365.25 + phase) +
              :rand.normal(0, 1)
          end)

        DataFrame.new(%{"ds" => dates, "y" => y, "id" => List.duplicate(id, length(dates))})
      end)
      |> DataFrame.concat_rows()
    end

    defp local_model(series) do
      Soothsayer.new(%{
        epochs: 30,
        seed: 1,
        trend: %{changepoints: 0},
        seasonality: %{weekly: %{enabled: false}},
        series: Map.merge(%{column: "id"}, series)
      })
    end

    defp panel_mae(fitted, holdout) do
      predictions = Soothsayer.predict(fitted, DataFrame.select(holdout, ["ds", "id"]))
      Series.subtract(predictions["yhat"], holdout["y"]) |> Series.abs() |> Series.mean()
    end

    test "local kernels fit series that share nothing but the model, shared ones can't" do
      start_date = ~D[2020-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      future_dates = Date.range(~D[2023-01-01], ~D[2023-03-31]) |> Enum.to_list()
      training = diverging_panel(dates, start_date, 1)
      holdout = diverging_panel(future_dates, start_date, 2)

      shared = Soothsayer.fit(local_model(%{}), training)
      local = Soothsayer.fit(local_model(%{trend: :local, seasonality: :local}), training)

      assert panel_mae(local, holdout) < panel_mae(shared, holdout) * 0.2

      # one trend kernel per series, with opposite slopes
      weights = Soothsayer.Trend.get_weights(local)
      assert Map.keys(weights) == ["a", "b"]
      assert Nx.to_number(weights["a"].kernel[0][0]) > 0
      assert Nx.to_number(weights["b"].kernel[0][0]) < 0
      assert Nx.shape(local.params.data["yearly_dense"]["kernel"]) == {2, 12, 1}
      assert Nx.shape(Soothsayer.Trend.get_weights(shared).kernel) == {1, 1}
    end

    test "local regularization pulls the series' kernels toward each other" do
      start_date = ~D[2020-01-01]
      dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      training = diverging_panel(dates, start_date, 3)

      spread = fn fitted ->
        weights = Soothsayer.Trend.get_weights(fitted)
        abs(Nx.to_number(weights["a"].kernel[0][0]) - Nx.to_number(weights["b"].kernel[0][0]))
      end

      free = Soothsayer.fit(local_model(%{trend: :local}), training)
      pulled = Soothsayer.fit(local_model(%{trend: :local, local_regularization: 10.0}), training)

      assert spread.(pulled) < spread.(free) * 0.5
    end

    test "local modes need a column and the regularization needs something local" do
      assert_raise ArgumentError, ~r/only be :local with a column/, fn ->
        Soothsayer.new(%{series: %{trend: :local}})
      end

      assert_raise ArgumentError, ~r/needs a :local trend or seasonality/, fn ->
        Soothsayer.new(%{series: %{column: "id", local_regularization: 1.0}})
      end

      assert_raise ArgumentError, ~r/series must be/, fn ->
        Soothsayer.new(%{series: %{column: "id", seasonality: :each}})
      end
    end
  end

  describe "future_timestamps/3" do
    test "continues from the last observation at the model's frequency, in the ds dtype" do
      dates = Date.range(~D[2022-01-01], ~D[2022-12-31]) |> Enum.to_list()
      frame = DataFrame.new(%{"ds" => dates, "y" => Enum.map(dates, &Date.day_of_year/1)})
      fitted = Soothsayer.fit(Soothsayer.new(%{epochs: 1}), frame)

      future = Soothsayer.future_timestamps(fitted, 3)
      assert Series.dtype(future) == :date
      assert Series.to_list(future) == [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]

      with_history = Soothsayer.future_timestamps(fitted, 2, include_history: true)
      assert Series.size(with_history) == 367
      assert Series.first(with_history) == ~D[2022-01-01]
      assert Series.last(with_history) == ~D[2023-01-02]

      # monthly data steps by months, naive datetimes stay naive datetimes
      months =
        Enum.map(0..23, &NaiveDateTime.new!(2020 + div(&1, 12), rem(&1, 12) + 1, 1, 0, 0, 0))

      monthly = DataFrame.new(%{"ds" => months, "y" => Enum.to_list(1..24)})

      monthly_fit =
        Soothsayer.fit(
          Soothsayer.new(%{epochs: 1, seasonality: %{weekly: %{enabled: false}}}),
          monthly
        )

      monthly_future = Soothsayer.future_timestamps(monthly_fit, 2)
      assert match?({:naive_datetime, _}, Series.dtype(monthly_future))

      assert Series.to_list(monthly_future) == [
               ~N[2022-01-01 00:00:00.000000],
               ~N[2022-02-01 00:00:00.000000]
             ]

      assert_raise ArgumentError, ~r/has not been fitted/, fn ->
        Soothsayer.future_timestamps(Soothsayer.new(), 3)
      end
    end

    test "gives every series its own block after its own last observation" do
      dates = Date.range(~D[2022-01-01], ~D[2022-03-31]) |> Enum.to_list()

      panel =
        DataFrame.concat_rows([
          DataFrame.new(%{
            "ds" => dates,
            "y" => Enum.map(dates, &Date.day_of_year/1),
            "id" => List.duplicate("a", 90)
          }),
          DataFrame.new(%{
            "ds" => Enum.take(dates, 60),
            "y" => Enum.to_list(1..60),
            "id" => List.duplicate("b", 60)
          })
        ])

      fitted = Soothsayer.fit(Soothsayer.new(%{series: %{column: "id"}, epochs: 1}), panel)
      future = Soothsayer.future_timestamps(fitted, 2)

      assert DataFrame.to_columns(future, atom_keys: true) == %{
               ds: [~D[2022-04-01], ~D[2022-04-02], ~D[2022-03-02], ~D[2022-03-03]],
               id: ["a", "a", "b", "b"]
             }

      assert DataFrame.n_rows(Soothsayer.predict(fitted, future)) == 4
    end
  end

  describe "loss option" do
    test "fits with :mae and with a custom elementwise loss, and refuses anything else" do
      dates = Date.range(~D[2022-01-01], ~D[2022-06-30]) |> Enum.to_list()

      df =
        DataFrame.new(%{"ds" => dates, "y" => Enum.map(dates, &(10 + Date.day_of_year(&1) / 10))})

      quiet = %{
        trend: %{changepoints: 0},
        seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
        epochs: 3
      }

      for loss <- [
            :mae,
            :mse,
            fn targets, predictions -> Nx.abs(Nx.subtract(targets, predictions)) end
          ] do
        fitted = Soothsayer.fit(Soothsayer.new(Map.put(quiet, :loss, loss)), df)
        assert fitted.config.loss == loss

        assert DataFrame.n_rows(
                 Soothsayer.predict(fitted, Soothsayer.future_timestamps(fitted, 3))
               ) == 3
      end

      assert_raise ArgumentError, ~r/loss must be :huber, :mae, :mse or a function/, fn ->
        Soothsayer.new(%{loss: :rmse})
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
