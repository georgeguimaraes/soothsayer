defmodule Soothsayer.BacktestTest do
  use ExUnit.Case, async: true

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Backtest

  describe "split/2" do
    test "holds out the last trunc(rows * fraction) rows, at least one" do
      data =
        DataFrame.new(%{
          "ds" => Enum.map(1..25, &Date.add(~D[2023-01-01], &1)),
          "y" => Enum.to_list(1..25)
        })

      {train, validation} = Backtest.split(data, 0.1)
      assert DataFrame.n_rows(train) == 23
      assert DataFrame.n_rows(validation) == 2

      {_, single} = Backtest.split(DataFrame.head(data, 5), 0.1)
      assert DataFrame.n_rows(single) == 1
    end
  end

  describe "run/3" do
    test "forecasts every validation date from each of the horizon origins before it" do
      :rand.seed(:exsss, {6, 6, 6})
      dates = Date.range(~D[2022-01-01], ~D[2022-12-31]) |> Enum.to_list()

      {y, _} =
        Enum.map_reduce(dates, 0.0, fn _, previous ->
          value = 0.7 * previous + :rand.normal(0, 1)
          {50 + value, value}
        end)

      data = DataFrame.new(%{"ds" => dates, "y" => y})

      model =
        Soothsayer.new(%{
          trend: %{enabled: false, changepoints: 0},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          ar: %{enabled: true, lags: 3, forecast_steps: 2},
          epochs: 3,
          seed: 1
        })

      result = Soothsayer.backtest(model, data, validation_fraction: 0.1)

      # 36 validation rows: origins at the last training date and the first 35
      # validation dates each forecast 2 steps, the last origin only 1
      validation_rows = 36
      assert DataFrame.n_rows(result.predictions) == 2 * validation_rows - 1

      assert result.predictions["step"] |> Series.distinct() |> Series.to_list() |> Enum.sort() ==
               [1, 2]

      assert Map.keys(result.by_step) |> Enum.sort() == [1, 2]
      assert result.model.params != nil

      # Every validation date appears once per step that can reach it
      counts = result.predictions["ds"] |> Series.frequencies()
      assert counts["counts"] |> Series.max() == 2

      # Overall metrics agree with a recomputation from the predictions frame
      recomputed = Backtest.metrics(result.predictions["y"], result.predictions["yhat"])
      assert_in_delta result.metrics.mean_absolute_error, recomputed.mean_absolute_error, 1.0e-6

      assert_in_delta result.metrics.root_mean_squared_error,
                      recomputed.root_mean_squared_error,
                      1.0e-6

      # Two steps ahead is harder than one on an AR process
      assert result.by_step[2].mean_absolute_error > result.by_step[1].mean_absolute_error
    end

    test "without auto-regression the horizon defaults to 1 and forecasts don't depend on the origin" do
      dates = Date.range(~D[2022-01-01], ~D[2022-06-30]) |> Enum.to_list()

      data =
        DataFrame.new(%{"ds" => dates, "y" => Enum.map(dates, &(Date.day_of_year(&1) * 1.0))})

      result =
        Soothsayer.backtest(
          Soothsayer.new(%{epochs: 2, seed: 1, trend: %{changepoints: 0}}),
          data
        )

      assert Map.keys(result.by_step) == [1]
      assert DataFrame.n_rows(result.predictions) == 18
    end
  end
end
