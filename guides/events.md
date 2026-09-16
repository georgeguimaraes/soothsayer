# Events

Events capture the impact of special occasions that affect your time series - holidays, promotions, product launches, and other one-off or recurring occurrences. Country holidays come built in through the [holidefs](https://hex.pm/packages/holidefs) package, see [Country Holidays](#country-holidays).

This is useful for:
- Holiday effects (Christmas, Black Friday)
- Marketing promotions and sales
- Product launches or announcements
- Any known future occurrence with measurable impact

## How It Works

The events component adds a spike or dip on specific dates:

```
events(t) = sum(z_e * e(t))
```

Where:
- `e(t)` = binary indicator (1 if event occurs, 0 otherwise)
- `z_e` = learned coefficient for each event

For more details, see [NeuralProphet's Events documentation](https://neuralprophet.com/html/events.html).

## Configuration

Events require two parts:

1. **Model config** - Define which events to model and their windows
2. **Events DataFrame** - Specify when each event occurs

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

### Event Config Parameters

| Parameter | Description |
|-----------|-------------|
| `steps_before` | Steps before the event that the effect covers, default `0` |
| `steps_after` | Steps after the event that the effect covers, default `0` |

Both are counts, so `%{steps_before: 2, steps_after: 1}` covers two steps before, the event itself and one step after, four features. `%{}` means the event date alone.

A step is one row of the data at its frequency: a day for daily data, an hour for hourly data. Event dates given as plain dates mean midnight, so on hourly data an event on `~D[2023-11-24]` with `steps_before: 1, steps_after: 1` covers 23:00 the day before, midnight and 01:00.

## Event Windows

Windows allow events to affect surrounding days, not just the event date itself.

### Simple Event (No Window)

For events that only affect the exact date:

```elixir
events: %{
  "sale" => %{steps_before: 0, steps_after: 0}
}
```

Creates 1 feature: `sale_0`

### Pre-Event Effects

For events where the impact starts before the date:

```elixir
events: %{
  "black_friday" => %{steps_before: 2, steps_after: 0}
}
```

Creates 3 features: `black_friday_-2`, `black_friday_-1`, `black_friday_0`

### Post-Event Effects

For events with lingering effects:

```elixir
events: %{
  "christmas" => %{steps_before: 0, steps_after: 2}
}
```

Creates 3 features: `christmas_0`, `christmas_+1`, `christmas_+2`

### Combined Windows

For events with both pre and post effects:

```elixir
events: %{
  "product_launch" => %{steps_before: 3, steps_after: 7}
}
```

Creates 11 features (-3 to +7), each learning its own coefficient.

## Example: Sales Events

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

## Prediction with Events

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

Without `events:` the remembered occurrences, the yearly recurring events and the country holidays still apply; only occurrences the model has never seen are left out.

## Inspecting Event Effects

Use `Soothsayer.get_event_effects/1` to see the learned impact of each event:

```elixir
effects = Soothsayer.get_event_effects(fitted)
# => %{"sale_0" => 48.5}

# For windowed events:
# => %{"black_friday_-1" => 12.3, "black_friday_0" => 45.2, "black_friday_+1" => 8.1}
```

Positive values indicate the event increases the forecast; negative values decrease it.

## Multiple Events

You can model multiple different events:

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

Each event type learns independent coefficients.

## Recurring Events

The same event can occur multiple times. Listing every occurrence works, and the model learns a single coefficient per event (per window position) applied to all of them:

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

## Country Holidays

Every holiday of a country becomes an event of its own, the way NeuralProphet's `add_country_holidays` works. The dates come from the [holidefs](https://hex.pm/packages/holidefs) package, an optional dependency, so add it to your project first:

```elixir
{:holidefs, "~> 0.4"}
```

Then name the countries and, optionally, one window for all of their holidays:

```elixir
model = Soothsayer.new(%{
  holidays: %{countries: [:us], steps_before: 1, steps_after: 1}
})

fitted = Soothsayer.fit(model, df)
```

That is all. Fit generates the holiday dates for the years in your data, prediction generates them for the years being forecast, and the fitted `config.events` lists the holidays next to your own events:

```elixir
Soothsayer.get_event_effects(fitted)
# => %{
#   "Christmas Day_-1" => 12.1, "Christmas Day_0" => 48.5, "Christmas Day_+1" => -3.2,
#   "Independence Day_-1" => ..., "Thanksgiving_0" => ..., ...
# }
```

Holidays are named as holidefs names them in English ("Independence Day", "Thanksgiving", "Christmas Day"), whatever Gettext locale your process has set. The same name from two countries is one event, so `countries: [:us, :gb]` has a single "Christmas Day". Your own events can't reuse a holiday name; fit raises if they do.

| Option | Description |
|--------|-------------|
| `countries` | holidefs locale codes, `:us`, `:gb`, `:br`, `:de`, ... Ask `Soothsayer.Holidays.supported/0` for the list. |
| `steps_before`, `steps_after` | One window for every holiday, steps before and after like event windows. Default `0`. |
| `regions` | holidefs regions such as `["us_ca"]`, added to the national holidays. |
| `include_informal` | Include holidays holidefs marks informal, like Good Friday in the US. Default `false`. |

A holiday is a plain date, so on hourly data it lands on midnight like any date event; use the window to cover the rest of the day. NeuralProphet's holiday regularization and multiplicative mode aren't there yet.

## Network Architecture

Events add an input branch to the network:

```elixir
# Each event with window creates multiple binary features
# e.g., "sale" with window -1 to +1 = 3 features

events_input_shape = {nil, positions, n_event_features}
```

`positions` is the number of timestamps in a training sample (one without auto-regression, the lags plus the forecast steps with it). The `events_dense` layer learns one weight per event feature, shared across positions.

## Next Steps

- [Trends](trends.md) - Piecewise linear trends with changepoints
- [Seasonality](seasonality.md) - Yearly and weekly patterns
- [Auto-Regression](autoregression.md) - Dependencies on recent values
- [The Basics](basics.md) - Fundamental concepts
