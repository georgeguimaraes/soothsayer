alias Explorer.DataFrame

# Load the data
df = DataFrame.from_csv!("toiletpaper_daily_sales.csv")

# Create a new Soothsayer model
model = Soothsayer.new()

# Fit the model
{model, metrics} = Soothsayer.fit(model, df, "D")

# Generate predictions
forecast = Soothsayer.predict(model, df)

# Display results
IO.inspect(metrics, label: "Metrics")
IO.inspect(forecast, label: "Forecast")
