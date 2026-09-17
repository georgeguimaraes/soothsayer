# Events

An event is a date you know about in advance that moves the series: a promotion, a product launch, a holiday. The model learns how much each one moves it. Country holidays come built in through [dayoff](https://hex.pm/packages/dayoff), see [Country Holidays](#country-holidays).

## How it works

The events component adds a spike or dip on the dates you name:

```
events(t) = sum(z_e * e(t))
```

Where:
- `e(t)` = binary indicator (1 if event occurs, 0 otherwise)
- `z_e` = learned coefficient for each event

For more details, see [NeuralProphet's Events documentation](https://neuralprophet.com/html/events.html).

## Configuration

An event needs two things: a window in the model config, and its dates in an events DataFrame passed to fit.

```elixir
alias Explorer.DataFrame

# 1. Configure the model
model = Soothsayer.new(%{
  events: %{
    "black_friday" => %{steps_before: 1, steps_after: 1},
    "christmas" => %{steps_before: 3, steps_after: 0}
  }
})

# 2. Create events DataFrame
events_df = DataFrame.new(%{
  "event" => ["black_friday", "black_friday", "christmas", "christmas"],
  "ds" => [~D[2022-11-25], ~D[2023-11-24], ~D[2022-12-25], ~D[2023-12-25]]
})

# 3. Fit with events
fitted = Soothsayer.fit(model, df, events: events_df)
```

With [several series](series.md) in one model the events frame may also carry the id column, and then each row is an event for that series only.

### Event config parameters

| Parameter | Description |
|-----------|-------------|
| `steps_before` | Steps before the event that the effect covers, default `0` |
| `steps_after` | Steps after the event that the effect covers, default `0` |
| `mode` | `:additive` (default) adds the effect to the forecast, `:multiplicative` scales it with the trend, see [Multiplicative events](#multiplicative-events) |
| `regularization` | L1 penalty on the event's coefficients, `nil` (default) or a number, see [Regularization](#regularization) |
| `recurring` | `:yearly` to repeat the event on its month and day every year, see [Recurring events](#recurring-events) |

Both windows are counts, so `%{steps_before: 2, steps_after: 1}` covers two steps before, the event itself and one step after, four features. `%{}` means the event date alone.

A step is one row of the data at its frequency: a day for daily data, an hour for hourly data. Event dates given as plain dates mean midnight, so on hourly data an event on `~D[2023-11-24]` with `steps_before: 1, steps_after: 1` covers 23:00 the day before, midnight and 01:00.

## Event windows

A window lets the effect spill over the days around the event.

### Simple event (no window)

Only the date itself:

```elixir
events: %{
  "sale" => %{steps_before: 0, steps_after: 0}
}
```

One feature, `sale_0`.

### Pre-event effects

The impact starts before the date:

```elixir
events: %{
  "black_friday" => %{steps_before: 2, steps_after: 0}
}
```

Three features: `black_friday_-2`, `black_friday_-1`, `black_friday_0`.

### Post-event effects

The impact lingers after the date:

```elixir
events: %{
  "christmas" => %{steps_before: 0, steps_after: 2}
}
```

Three features: `christmas_0`, `christmas_+1`, `christmas_+2`.

### Combined windows

Both sides:

```elixir
events: %{
  "product_launch" => %{steps_before: 3, steps_after: 7}
}
```

Eleven features, `-3` to `+7`, each with its own coefficient.

## Example: sales events

```elixir
alias Explorer.DataFrame
alias Explorer.Series

# Generate data with sale spikes
:rand.seed(:exsss, {42, 42, 42})

dates = Enum.map(0..364, fn i -> Date.add(~D[2023-01-01], i) end)
sale_dates = [~D[2023-03-15], ~D[2023-09-15]]

y = Enum.map(dates, fn date ->
  trend = 100 + 0.1 * Date.diff(date, ~D[2023-01-01])
  spike = if date in sale_dates, do: 50, else: 0
  noise = :rand.normal(0, 5)
  trend + spike + noise
end)

df = DataFrame.new(%{"ds" => dates, "y" => y})

events_df = DataFrame.new(%{
  "event" => ["sale", "sale"],
  "ds" => sale_dates
})

# Fit model
model = Soothsayer.new(%{
  trend: %{enabled: true, changepoints: 0},
  seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
  events: %{
    "sale" => %{steps_before: 0, steps_after: 0}
  },
  epochs: 50
})

fitted = Soothsayer.fit(model, df, events: events_df)
```

## Prediction with events

The model remembers the occurrences it was fitted with, so predicting inside the training period needs nothing extra. For a one-off occurrence in the future, pass it:

```elixir
# Future dates
future_dates = Series.from_list([~D[2024-01-01], ~D[2024-01-02], ~D[2024-03-15]])

# Future events
future_events = DataFrame.new(%{
  "event" => ["sale"],
  "ds" => [~D[2024-03-15]]
})

predictions = Soothsayer.predict(fitted, future_dates, events: future_events)
```

Without `events:` the remembered occurrences, the yearly recurring events and the country holidays still apply. Only occurrences the model has never seen are left out.

## Inspecting event effects

`Soothsayer.get_event_effects/1` returns the learned coefficient of every event feature:

```elixir
effects = Soothsayer.get_event_effects(fitted)
# => %{"sale_0" => 48.5}

# For windowed events:
# => %{"black_friday_-1" => 12.3, "black_friday_0" => 45.2, "black_friday_+1" => 8.1}
```

A positive coefficient lifts the forecast on that day, a negative one lowers it. Coefficients are per normalized unit of the event feature (inputs are z-scored like everything else the network sees), so read them relative to each other rather than in units of `y`. The `events` column of `Soothsayer.predict/3` has the combined effect per date in the units of `y`.

## Multiplicative events

By default an event adds a fixed amount. When the bump grows with the level of the series, a promo that lifts sales by 30% whatever the sales are that year, make it multiplicative:

```elixir
model = Soothsayer.new(%{
  events: %{"promo" => %{steps_before: 1, steps_after: 1, mode: :multiplicative}}
})
```

The coefficient is then a fraction of the trend per normalized unit of the event feature, and the effect on a date is that fraction times the trend on that date. Additive and multiplicative events can share a model. Each mode gets its own layer (`events_dense` and `events_multiplicative_dense`) and predict still reports one `events` column with everything in the units of `y`, so the components keep adding up to `yhat`. This is NeuralProphet's multiplicative events, with one difference: the trend that scales the effect is detached only at the lag positions of an auto-regressive sample, NeuralProphet detaches it everywhere.

With the trend disabled the scale is just the level of the series, so a multiplicative event behaves like an additive one with a different unit. NeuralProphet raises in that case, soothsayer lets it through.

## Regularization

An L1 penalty on an event's coefficients pulls the ones the data doesn't support toward zero, handy when you list many events and only some of them matter:

```elixir
model = Soothsayer.new(%{
  events: %{
    "black_friday" => %{steps_before: 2, steps_after: 1},
    "maybe_relevant" => %{steps_before: 0, steps_after: 0, regularization: 0.5}
  }
})
```

The penalty is the lambda times the sum of the absolute coefficients of that event, applied from the first training step, same as the `regularization` on `ar` and `trend`. NeuralProphet applies its event penalties only in the last third of training and rescales some of its lambdas, so its values don't carry over. `0` and `nil` both mean no penalty.

## Multiple events

Any number of events can go in the same model:

```elixir
events_df = DataFrame.new(%{
  "event" => ["black_friday", "black_friday", "christmas", "christmas", "new_year"],
  "ds" => [~D[2022-11-25], ~D[2023-11-24], ~D[2022-12-25], ~D[2023-12-25], ~D[2024-01-01]]
})

model = Soothsayer.new(%{
  events: %{
    "black_friday" => %{steps_before: 2, steps_after: 1},
    "christmas" => %{steps_before: 7, steps_after: 0},
    "new_year" => %{steps_before: 0, steps_after: 0}
  }
})
```

Each one gets its own coefficients.

## Recurring events

The same event can happen many times. List every occurrence and the model learns one coefficient per window position, shared by all of them:

```elixir
events_df = DataFrame.new(%{
  "event" => ["christmas", "christmas", "christmas"],
  "ds" => [~D[2021-12-25], ~D[2022-12-25], ~D[2023-12-25]]
})
```

For something that falls on the same month and day every year, say so once with `recurring: :yearly` and give one occurrence:

```elixir
model = Soothsayer.new(%{
  events: %{"founders_day" => %{steps_before: 0, steps_after: 1, recurring: :yearly}}
})

events_df = DataFrame.new(%{"event" => ["founders_day"], "ds" => [~D[2022-05-10]]})
fitted = Soothsayer.fit(model, df, events: events_df)
```

Every year in the training data gets its May 10, and so does every year you predict, with nothing passed at predict. A February 29 recurs in leap years only, and a naive datetime keeps its time of day. For holidays that move around the calendar, use country holidays.

## Country holidays

Every holiday of a country becomes an event of its own, like NeuralProphet's `add_country_holidays`. The dates come from [dayoff](https://hex.pm/packages/dayoff), which ships the date-holidays dataset for 200+ countries, states and regions. Name the countries and, if you want, one window for all of their holidays:

```elixir
model = Soothsayer.new(%{
  holidays: %{countries: ["US"], steps_before: 1, steps_after: 1}
})

fitted = Soothsayer.fit(model, df)
```

Fit generates the holiday dates for the years in your data, predict does the same for the years it forecasts, and the fitted `config.events` lists the holidays next to your own events:

```elixir
Soothsayer.get_event_effects(fitted)
# => %{
#   "Christmas Day_-1" => 12.1, "Christmas Day_0" => 48.5, "Christmas Day_+1" => -3.2,
#   "Independence Day_-1" => ..., "Thanksgiving Day_0" => ..., ...
# }
```

Holidays are named in English by default ("Independence Day", "Thanksgiving Day", "Christmas Day"). The same name from two countries is one event, so `countries: ["US", "GB"]` has a single "Christmas Day". A substitute day ("Christmas Day (substitute day)") is its own event, since the observed Monday behaves differently from the 25th. Your own events can't reuse a holiday name, fit raises if they do.

| Option | Description |
|--------|-------------|
| `countries` | dayoff codes: `"US"`, `:br`, or with a state or region, `"US-CA"`, `"DE-BY-A"`. `Soothsayer.Holidays.supported/0` lists the countries, `Dayoff.states/1` and `Dayoff.regions/2` the subdivisions. |
| `steps_before`, `steps_after` | One window for every holiday, steps before and after like event windows. Default `0`. |
| `types` | Which dayoff holiday types count: `:public`, `:bank`, `:school`, `:optional`, `:observance`. Default `[:public]`. |
| `language` | Language of the holiday names, which are the event names. Default `"en"`, falling back to the country's own language when a name has no translation. |
| `mode` | `:additive` (default) or `:multiplicative`, for every holiday at once, like an event's `mode`. |
| `regularization` | One L1 lambda for every holiday, like an event's `regularization`. Default `nil`. |

A holiday is a plain date, so on hourly data it lands on midnight like any date event. Use the window to cover the rest of the day. Holidays share one window, one mode and one regularization, like NeuralProphet's `add_country_holidays`, while an event you name yourself can have its own.

## Network architecture

Events add an input branch to the network:

```elixir
# "sale" with steps_before: 1, steps_after: 1 is 3 features
events_input_shape = {nil, positions, n_event_features}
```

`positions` is the number of timestamps in a training sample (one without auto-regression, the lags plus the forecast steps with it). The feature columns are laid out additive events first, then multiplicative, each group sorted by name. The `events_dense` layer learns one weight per additive feature and `events_multiplicative_dense` one per multiplicative feature, shared across positions, and the multiplicative output is multiplied by the trend before the two are summed.

## Next steps

- [Trends](trends.md): piecewise linear trends with changepoints
- [Seasonality](seasonality.md): yearly and weekly patterns
- [Auto-Regression](autoregression.md): dependence on recent values
- [The Basics](basics.md): data format, fit, predict
