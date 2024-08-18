defmodule SoothsayerTest do
  use ExUnit.Case
  doctest Soothsayer

  alias Explorer.DataFrame
  alias Explorer.Series

  describe "Soothsayer" do
    test "simple linear regression" do
      # Create sample data
      x = Enum.to_list(1..100)
      y = Enum.map(x, fn x -> 2 * x + 1 + :rand.normal(0, 0.1) end)
      df = DataFrame.new(%{"x" => x, "y" => y})

      # Create and fit the model
      model = Soothsayer.new()
      fitted_model = Soothsayer.fit(model, df)

      # Make predictions
      x_test = Series.from_list([101, 102, 103])
      predictions = Soothsayer.predict(fitted_model, x_test)

      # Check predictions
      predictions_list = Nx.to_flat_list(predictions)
      assert length(predictions_list) == 3

      # Check if predictions are close to expected values
      Enum.zip(predictions_list, [203, 205, 207])
      |> Enum.each(fn {pred, expected} ->
        assert_in_delta pred, expected, 5
      end)
    end
  end
end
