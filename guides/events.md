# Events

Events capture the impact of special occasions that affect your time series - holidays, promotions, product launches, and other one-off or recurring occurrences.

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
    "black_friday" => %{lower_window: -1, upper_window: 1},
    "christmas" => %{lower_window: -3, upper_window: 0}
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
| `lower_window` | Days before event (use negative numbers) |
| `upper_window` | Days after event (use positive numbers) |

## Event Windows

Windows allow events to affect surrounding days, not just the event date itself.

### Simple Event (No Window)

For events that only affect the exact date:

```elixir
events: %{
  "sale" => %{lower_window: 0, upper_window: 0}
}
```

Creates 1 feature: `sale_0`

### Pre-Event Effects

For events where the impact starts before the date:

```elixir
events: %{
  "black_friday" => %{lower_window: -2, upper_window: 0}
}
```

Creates 3 features: `black_friday_-2`, `black_friday_-1`, `black_friday_0`

### Post-Event Effects

For events with lingering effects:

```elixir
events: %{
  "christmas" => %{lower_window: 0, upper_window: 2}
}
```

Creates 3 features: `christmas_0`, `christmas_+1`, `christmas_+2`

### Combined Windows

For events with both pre and post effects:

```elixir
events: %{
  "product_launch" => %{lower_window: -3, upper_window: 7}
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
    "sale" => %{lower_window: 0, upper_window: 0}
  },
  epochs: 50
})

fitted = Soothsayer.fit(model, df, events: events_df)
```

## Prediction with Events

When predicting, you must provide events for the prediction dates:

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

If you omit the `events:` option during prediction, the model won't include event effects.

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
    "black_friday" => %{lower_window: -2, upper_window: 1},
    "christmas" => %{lower_window: -7, upper_window: 0},
    "new_year" => %{lower_window: 0, upper_window: 0}
  }
})
```

Each event type learns independent coefficients.

## Recurring Events

The same event can occur multiple times (e.g., annual holidays). Just include all occurrences in the events DataFrame:

```elixir
events_df = DataFrame.new(%{
  "event" => ["christmas", "christmas", "christmas"],
  "ds" => [~D[2021-12-25], ~D[2022-12-25], ~D[2023-12-25]]
})
```

The model learns a single coefficient per event (per window position), applied to all occurrences.

## Network Architecture

Events add an input branch to the network:

```elixir
# Each event with window creates multiple binary features
# e.g., "sale" with window -1 to +1 = 3 features

events_input_shape = {nil, n_event_features}
```

The `events_dense` layer learns weights for all event features.

## Next Steps

- [Trends](trends.md) - Piecewise linear trends with changepoints
- [Seasonality](seasonality.md) - Yearly and weekly patterns
- [Auto-Regression](autoregression.md) - Dependencies on recent values
- [The Basics](basics.md) - Fundamental concepts
