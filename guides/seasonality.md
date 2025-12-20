# Seasonality

Seasonality captures repeating patterns at fixed intervals. Soothsayer supports yearly and weekly seasonality using Fourier terms.

This is useful for:
- Holiday shopping peaks (yearly)
- Summer/winter demand patterns (yearly)
- Weekend dips in activity (weekly)
- Monday peaks in emails (weekly)

## How It Works

Seasonality is modeled using Fourier series:

```
seasonality(t) = sum(a_n * cos(2*pi*n*t/P) + b_n * sin(2*pi*n*t/P))
```

Where:
- `P` = period (365.25 days for yearly, 7 days for weekly)
- `n` = Fourier term index (1 to N)
- `a_n`, `b_n` = learned coefficients

More Fourier terms allow more complex seasonal shapes.

For more details, see [NeuralProphet's Seasonality documentation](https://neuralprophet.com/html/seasonal-modeling.html).

## Configuration

```elixir
model = Soothsayer.new(%{
  seasonality: %{
    yearly: %{
      enabled: true,      # Enable yearly seasonality
      fourier_terms: 6    # Number of Fourier terms
    },
    weekly: %{
      enabled: true,      # Enable weekly seasonality
      fourier_terms: 3    # Number of Fourier terms
    }
  }
})
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yearly.enabled` | `true` | Enable yearly (annual) patterns |
| `yearly.fourier_terms` | `6` | Flexibility of yearly pattern |
| `weekly.enabled` | `true` | Enable weekly patterns |
| `weekly.fourier_terms` | `3` | Flexibility of weekly pattern |

## Yearly Seasonality

Captures patterns that repeat every year:

```elixir
model = Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: true, fourier_terms: 6},
    weekly: %{enabled: false}
  }
})
```

Good for:
- Holiday shopping peaks
- Summer vacation patterns
- Agricultural cycles
- Weather-driven demand

### Fourier Terms for Yearly

| fourier_terms | Flexibility | Use when |
|---------------|-------------|----------|
| 3 | Low | Smooth, simple patterns (one peak per year) |
| 6 | Medium | Most cases (default) |
| 10+ | High | Complex patterns with multiple peaks |

## Weekly Seasonality

Captures patterns that repeat every week:

```elixir
model = Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: false},
    weekly: %{enabled: true, fourier_terms: 3}
  }
})
```

Good for:
- Weekend vs weekday differences
- Day-of-week patterns
- Business cycle effects

### Fourier Terms for Weekly

| fourier_terms | Flexibility | Use when |
|---------------|-------------|----------|
| 2 | Low | Simple weekday/weekend split |
| 3 | Medium | Most cases (default) |
| 5+ | High | Complex day-specific patterns |

## Choosing Fourier Terms

More terms = more flexibility = more risk of overfitting.

**Signs you need more terms:**
- Residuals show repeating patterns
- Known complex seasonal shape (multiple peaks per cycle)

**Signs you need fewer terms:**
- Model fits noise in training but not test data
- Seasonal pattern looks jagged or implausible

Start with defaults and adjust based on results.

## Example: Extracting Seasonal Components

```elixir
alias Explorer.DataFrame
alias Explorer.Series

# Generate data with both yearly and weekly seasonality
start_date = ~D[2020-01-01]
dates = Date.range(start_date, ~D[2022-12-31])

y = Enum.map(dates, fn date ->
  days = Date.diff(date, start_date)
  trend = 1000
  yearly = 50 * :math.sin(2 * :math.pi() * days / 365.25)
  weekly = 20 * :math.cos(2 * :math.pi() * Date.day_of_week(date) / 7)
  noise = :rand.normal(0, 10)
  trend + yearly + weekly + noise
end)

df = DataFrame.new(%{"ds" => dates, "y" => y})

# Fit model
model = Soothsayer.new(%{
  trend: %{changepoints: 0},  # Simple linear trend
  seasonality: %{
    yearly: %{enabled: true, fourier_terms: 6},
    weekly: %{enabled: true, fourier_terms: 3}
  },
  epochs: 50
})

fitted = Soothsayer.fit(model, df)

# Extract components
components = Soothsayer.predict_components(fitted, df["ds"])

# components.yearly_seasonality contains the yearly pattern
# components.weekly_seasonality contains the weekly pattern
```

## Disabling Seasonality

If your data has no seasonal patterns:

```elixir
model = Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: false},
    weekly: %{enabled: false}
  }
})
```

Or disable just one type:

```elixir
# Only yearly seasonality
model = Soothsayer.new(%{
  seasonality: %{
    yearly: %{enabled: true},
    weekly: %{enabled: false}
  }
})
```

## Data Requirements

- **Yearly seasonality**: Needs at least 2 years of data to learn reliably
- **Weekly seasonality**: Needs at least a few weeks of data

With less data, the model may learn noise rather than true patterns. Consider disabling seasonality if you don't have enough data.

## Network Architecture

Each seasonality type adds input features to the network:

```elixir
# Yearly: 2 * fourier_terms features (sin and cos)
yearly_features = 2 * 6  # => 12 features for fourier_terms: 6

# Weekly: 2 * fourier_terms features
weekly_features = 2 * 3  # => 6 features for fourier_terms: 3
```

## Next Steps

- [Auto-Regression](autoregression.md) - Capture dependencies on recent values
- [Trends](trends.md) - Piecewise linear trends with changepoints
