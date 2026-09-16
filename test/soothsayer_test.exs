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

      # 10 changepoints over the first 80% of 1096 days sit every 87.7 days,
      # the sixth at day 526, where the level jumps by 40.
      y =
        Enum.map(dates, fn date ->
          days = Date.diff(date, start_date)
          jump = if days >= 526, do: 40.0, else: 0.0
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
      assert_raise ArgumentError, ~r/trend.growth must be :linear or :discontinuous/, fn ->
        Soothsayer.new(%{trend: %{growth: :off}})
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

      assert length(fitted.config.training_data.timestamps) == 38

      refute Enum.at(dates, 5) in Enum.map(
               fitted.config.training_data.timestamps,
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

      assert length(fitted.config.training_data.timestamps) == 40

      assert_in_delta fitted.config.training_data.known_values[gap_timestamp],
                      (15.0 - mean) / std,
                      1.0e-4
    end

    test "with auto-regression a missing row is inserted on the grid" do
      {dates, data} = gapped_frame(40, [])
      without_row = DataFrame.filter_with(data, &Series.not_equal(&1["ds"], Enum.at(dates, 20)))
      assert DataFrame.n_rows(without_row) == 39

      fitted = Soothsayer.fit(ar_model(), without_row)
      assert length(fitted.config.training_data.timestamps) == 40
      assert fitted.config.training_data.known_values |> Map.keys() |> length() == 40
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
      known = fitted.config.training_data.known_values

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
