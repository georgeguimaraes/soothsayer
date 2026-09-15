defmodule Soothsayer.QuantilesTest do
  use ExUnit.Case, async: true

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Quantiles

  describe "pinball_loss/3" do
    test "penalizes under-prediction more for high quantiles" do
      targets = Nx.tensor([[10.0]])
      low = Nx.tensor([[8.0]])
      high = Nx.tensor([[12.0]])

      # For q = 0.9, missing low costs 0.9 * 2, missing high costs 0.1 * 2
      assert_in_delta Quantiles.pinball_loss(targets, low, 0.9) |> Nx.to_number(), 1.8, 1.0e-5
      assert_in_delta Quantiles.pinball_loss(targets, high, 0.9) |> Nx.to_number(), 0.2, 1.0e-5
    end
  end

  describe "config" do
    test "quantiles are sorted, deduplicated floats" do
      assert Soothsayer.new(%{quantiles: [0.9, 0.1, 0.9]}).config.quantiles == [0.1, 0.9]
    end

    test "rejects quantiles outside (0, 1) and non-lists" do
      assert_raise ArgumentError, ~r/strictly between 0 and 1/, fn ->
        Soothsayer.new(%{quantiles: [0.1, 1.0]})
      end

      assert_raise ArgumentError, ~r/must be a list/, fn -> Soothsayer.new(%{quantiles: 0.9}) end
    end

    test "no quantiles means no quantile outputs and an empty map" do
      dates = Date.range(~D[2023-01-01], ~D[2023-03-31]) |> Enum.to_list()
      df = DataFrame.new(%{"ds" => dates, "y" => Enum.map(dates, &(Date.day_of_year(&1) * 1.0))})
      fitted_model = Soothsayer.fit(Soothsayer.new(%{epochs: 1, trend: %{changepoints: 0}}), df)

      components = Soothsayer.predict_components(fitted_model, Series.from_list([~D[2023-04-01]]))
      assert components.quantiles == %{}
    end
  end

  describe "prediction intervals" do
    defp fit_with_quantiles(dates, y, overrides) do
      config =
        Map.merge(
          %{
            trend: %{changepoints: 0},
            seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
            quantiles: [0.1, 0.9],
            epochs: 60,
            seed: 21
          },
          overrides
        )

      Soothsayer.fit(Soothsayer.new(config), DataFrame.new(%{"ds" => dates, "y" => y}))
    end

    test "the 10/90 interval covers roughly 80% of held out points and brackets the median" do
      :rand.seed(:exsss, {4, 4, 4})
      start_date = ~D[2020-01-01]
      training_dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      holdout_dates = Date.range(~D[2023-01-01], ~D[2023-06-30]) |> Enum.to_list()
      series = fn date -> 100 + 0.1 * Date.diff(date, start_date) + :rand.normal(0, 5) end

      fitted_model = fit_with_quantiles(training_dates, Enum.map(training_dates, series), %{})
      holdout = Enum.map(holdout_dates, series)

      components = Soothsayer.predict_components(fitted_model, Series.from_list(holdout_dates))
      lower = Nx.to_flat_list(components.quantiles[0.1])
      upper = Nx.to_flat_list(components.quantiles[0.9])
      median = Nx.to_flat_list(components.combined)

      covered =
        [holdout, lower, upper]
        |> Enum.zip()
        |> Enum.count(fn {actual, low, high} -> actual >= low and actual <= high end)

      coverage = covered / length(holdout)
      assert coverage > 0.65 and coverage < 0.95, "coverage was #{coverage}"

      for {low, mid, high} <- Enum.zip([lower, median, upper]) do
        assert low <= mid and mid <= high
      end
    end

    test "interval width grows when the noise grows with the level" do
      :rand.seed(:exsss, {8, 8, 8})
      start_date = ~D[2020-01-01]
      training_dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      # Noise standard deviation is 5% of the level, so it triples over the series
      series = fn date ->
        level = 100 + 0.2 * Date.diff(date, start_date)
        level + :rand.normal(0, 0.05 * level)
      end

      fitted_model = fit_with_quantiles(training_dates, Enum.map(training_dates, series), %{})

      components =
        Soothsayer.predict_components(
          fitted_model,
          Series.from_list([~D[2020-03-01], ~D[2022-12-01]])
        )

      [early_width, late_width] =
        Nx.subtract(components.quantiles[0.9], components.quantiles[0.1]) |> Nx.to_flat_list()

      assert late_width > early_width * 1.5
    end

    test "works together with multi-step auto-regression" do
      :rand.seed(:exsss, {2, 2, 2})
      dates = Date.range(~D[2022-01-01], ~D[2023-12-31]) |> Enum.to_list()

      {y, _} =
        Enum.map_reduce(dates, 0.0, fn _date, previous ->
          value = 0.8 * previous + :rand.normal(0, 2)
          {50 + value, value}
        end)

      fitted_model =
        fit_with_quantiles(dates, y, %{
          trend: %{enabled: false, changepoints: 0},
          ar: %{enabled: true, lags: 3, forecast_steps: 3},
          epochs: 20
        })

      future = Series.from_list(Enum.map(1..6, &Date.add(List.last(dates), &1)))
      components = Soothsayer.predict_components(fitted_model, future)

      assert Nx.shape(components.quantiles[0.9]) == {6, 1}

      assert Nx.all(Nx.greater_equal(components.quantiles[0.9], components.combined))
             |> Nx.to_number() == 1
    end
  end
end
