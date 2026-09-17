defmodule Soothsayer.CrossValidationTest do
  use ExUnit.Case, async: false

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.CrossValidation

  doctest Soothsayer.CrossValidation

  defp daily(dates, level) do
    y = Enum.map(dates, fn date -> level + 0.1 * Date.diff(date, hd(dates)) end)
    DataFrame.new(%{"ds" => dates, "y" => y})
  end

  defp quiet_model(extra \\ %{}) do
    Soothsayer.new(
      Map.merge(
        %{
          trend: %{changepoints: 0},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          epochs: 2,
          seed: 1
        },
        extra
      )
    )
  end

  describe "cutoffs/4" do
    test "spaced period apart from the last one horizon before the end, while initial rows remain" do
      dates = Enum.map(0..99, &Date.add(~D[2024-01-01], &1))

      assert CrossValidation.cutoffs(dates, 10, 30, 30) ==
               [~D[2024-01-30], ~D[2024-02-29], ~D[2024-03-30]]

      # exactly one cutoff when initial + horizon fills the data
      assert CrossValidation.cutoffs(dates, 10, 30, 90) == [~D[2024-03-30]]

      assert_raise ArgumentError, ~r/at least initial \+ horizon rows/, fn ->
        CrossValidation.cutoffs(dates, 10, 30, 91)
      end
    end
  end

  describe "run/3" do
    test "refits at every cutoff and scores the horizon after it, by step" do
      dates = Enum.map(0..119, &Date.add(~D[2024-01-01], &1))
      df = daily(dates, 100)

      result = CrossValidation.run(quiet_model(), df, horizon: 5, period: 20, initial: 60)

      assert result.cutoffs == [~D[2024-03-15], ~D[2024-04-04], ~D[2024-04-24]]
      assert DataFrame.names(result.predictions) == ["cutoff", "ds", "step", "y", "yhat"]
      assert DataFrame.n_rows(result.predictions) == 15
      assert result.predictions["step"] |> Series.to_list() |> Enum.take(5) == [1, 2, 3, 4, 5]

      # the first forecast row of a cutoff is the row right after it
      first = DataFrame.head(result.predictions, 1)
      assert Series.first(first["cutoff"]) == ~D[2024-03-15]
      assert Series.first(first["ds"]) == ~D[2024-03-16]

      assert Map.keys(result.by_step) |> Enum.sort() == [1, 2, 3, 4, 5]

      assert Map.keys(result.metrics) |> Enum.sort() == [
               :mean_absolute_error,
               :mean_absolute_percentage_error,
               :root_mean_squared_error,
               :symmetric_mean_absolute_percentage_error
             ]

      assert result.metrics.mean_absolute_percentage_error > 0
    end

    test "takes explicit cutoffs, sorted, and refuses ones without rows around them" do
      dates = Enum.map(0..59, &Date.add(~D[2024-01-01], &1))
      df = daily(dates, 10)

      result =
        CrossValidation.run(quiet_model(), df,
          horizon: 3,
          cutoffs: [~D[2024-02-10], ~D[2024-01-20]]
        )

      assert result.cutoffs == [~D[2024-01-20], ~D[2024-02-10]]
      assert DataFrame.n_rows(result.predictions) == 6

      assert_raise ArgumentError, ~r/leaves no training rows/, fn ->
        CrossValidation.run(quiet_model(), df, horizon: 3, cutoffs: [~D[2024-02-29]])
      end
    end

    test "reports coverage with quantiles and keeps the interval columns" do
      dates = Enum.map(0..89, &Date.add(~D[2024-01-01], &1))
      df = daily(dates, 50)

      result =
        CrossValidation.run(quiet_model(%{quantiles: [0.1, 0.9]}), df,
          horizon: 4,
          period: 30,
          initial: 40
        )

      assert "yhat_10" in DataFrame.names(result.predictions)
      assert Map.has_key?(result.metrics, :coverage)
      assert result.by_step[4].coverage >= 0.0
    end

    test "shares each cutoff across several series and keeps their rows apart" do
      dates = Enum.map(0..89, &Date.add(~D[2024-01-01], &1))

      panel =
        DataFrame.concat_rows([
          DataFrame.put(daily(dates, 10), "id", Series.from_list(List.duplicate("a", 90))),
          DataFrame.put(
            daily(Enum.drop(dates, 10), 20),
            "id",
            Series.from_list(List.duplicate("b", 80))
          )
        ])

      result =
        CrossValidation.run(quiet_model(%{series: %{column: "id"}}), panel,
          horizon: 3,
          period: 30,
          initial: 40
        )

      assert DataFrame.names(result.predictions) |> Enum.take(3) == ["id", "cutoff", "ds"]
      assert DataFrame.n_rows(result.predictions) == length(result.cutoffs) * 2 * 3

      steps_by_id =
        result.predictions
        |> DataFrame.filter_with(&Series.equal(&1["cutoff"], hd(result.cutoffs)))
        |> DataFrame.to_columns()

      assert steps_by_id["step"] == [1, 2, 3, 1, 2, 3]
      assert steps_by_id["id"] == ["a", "a", "a", "b", "b", "b"]
    end
  end
end
