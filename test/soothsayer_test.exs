defmodule SoothsayerTest do
  use ExUnit.Case
  doctest Soothsayer

  alias Explorer.DataFrame

  test "basic workflow" do
    # Create sample data
    dates = Date.range(~D[2023-01-01], ~D[2023-01-10])
    values = Enum.map(1..10, fn x -> x * 2 end)
    df = DataFrame.new(%{"date" => dates, "y" => values})

    # Create and fit model
    model = Soothsayer.new()
    {fitted_model, metrics} = Soothsayer.fit(model, df, "D")

    assert is_map(metrics)
    assert Map.has_key?(metrics, :mse)
    assert Map.has_key?(metrics, :mae)

    # Generate predictions
    forecast = Soothsayer.predict(fitted_model, df)

    assert %DataFrame{} = forecast
    assert DataFrame.n_rows(forecast) == DataFrame.n_rows(df)
    assert "date" in DataFrame.names(forecast)
    assert "yhat" in DataFrame.names(forecast)
  end
end
