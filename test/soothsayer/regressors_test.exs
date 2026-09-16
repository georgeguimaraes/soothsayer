defmodule Soothsayer.RegressorsTest do
  use ExUnit.Case, async: true
  doctest Soothsayer.Regressors

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Regressors

  describe "build_features/3" do
    test "stacks regressor values per date in config order" do
      dataframe =
        DataFrame.new(%{
          "ds" => [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]],
          "temperature" => [20.0, 22.5, 19.0],
          "price" => [1, 2, 3]
        })

      result =
        Regressors.build_features([~D[2023-01-03], ~D[2023-01-01]], dataframe, [
          "price",
          "temperature"
        ])

      assert Nx.shape(result) == {2, 2}
      assert Nx.to_flat_list(result) == [3.0, 19.0, 1.0, 20.0]
    end

    test "raises when a date has no regressor row" do
      dataframe = DataFrame.new(%{"ds" => [~D[2023-01-01]], "temperature" => [20.0]})

      assert_raise ArgumentError, ~r/Regressor "temperature" has no value for 2023-01-02/, fn ->
        Regressors.build_features([~D[2023-01-01], ~D[2023-01-02]], dataframe, ["temperature"])
      end
    end

    test "raises when a regressor column is missing" do
      dataframe = DataFrame.new(%{"ds" => [~D[2023-01-01]], "temp" => [20.0]})

      assert_raise ArgumentError, ~r/Regressor column "temperature" not found/, fn ->
        Regressors.build_features([~D[2023-01-01]], dataframe, ["temperature"])
      end
    end
  end

  describe "fitting and predicting with regressors" do
    # y = trend + 3 * temperature + noise, where temperature is an independent
    # signal the date alone can't explain.
    defp regressor_data(dates, start_date) do
      temperature =
        Enum.map(dates, fn date ->
          days = Date.diff(date, start_date)
          20 + 8 * :math.sin(days / 23) + :rand.normal(0, 2)
        end)

      y =
        dates
        |> Enum.zip(temperature)
        |> Enum.map(fn {date, temp} ->
          100 + 0.05 * Date.diff(date, start_date) + 3 * temp + :rand.normal(0, 1)
        end)

      DataFrame.new(%{"ds" => dates, "y" => y, "temperature" => temperature})
    end

    defp base_config(overrides) do
      Map.merge(
        %{
          trend: %{changepoints: 0},
          seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
          epochs: 40,
          seed: 3
        },
        overrides
      )
    end

    test "a regressor the dates can't explain cuts holdout error and gets a positive coefficient" do
      :rand.seed(:exsss, {9, 9, 9})
      start_date = ~D[2021-01-01]
      training_dates = Date.range(start_date, ~D[2022-12-31]) |> Enum.to_list()
      holdout_dates = Date.range(~D[2023-01-01], ~D[2023-02-28]) |> Enum.to_list()

      training = regressor_data(training_dates, start_date)
      holdout = regressor_data(holdout_dates, start_date)

      mean_absolute_error = fn predictions ->
        predictions
        |> Nx.flatten()
        |> Nx.subtract(holdout["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32}))
        |> Nx.abs()
        |> Nx.mean()
        |> Nx.to_number()
      end

      without = Soothsayer.fit(Soothsayer.new(base_config(%{})), training)

      without_error =
        mean_absolute_error.(
          Soothsayer.predict(without, holdout["ds"])["yhat"]
          |> Series.to_tensor()
        )

      with_regressor =
        Soothsayer.fit(Soothsayer.new(base_config(%{regressors: ["temperature"]})), training)

      components =
        Soothsayer.predict_components(with_regressor, holdout["ds"], regressors: holdout)

      with_error = mean_absolute_error.(components.combined)

      assert with_error < without_error * 0.5

      assert %{"temperature" => coefficient} = Soothsayer.get_regressor_effects(with_regressor)
      assert coefficient > 0

      summed =
        [
          :trend,
          :yearly_seasonality,
          :weekly_seasonality,
          :daily_seasonality,
          :ar,
          :events,
          :regressors,
          :lagged_regressors
        ]
        |> Enum.map(&components[&1])
        |> Enum.reduce(&Nx.add/2)

      assert Nx.all_close(summed, components.combined, atol: 1.0e-2) |> Nx.to_number() == 1
    end

    test "predicting without the regressors dataframe raises" do
      :rand.seed(:exsss, {9, 9, 9})
      start_date = ~D[2022-01-01]
      dates = Date.range(start_date, ~D[2022-03-31]) |> Enum.to_list()
      training = regressor_data(dates, start_date)

      fitted_model =
        Soothsayer.fit(
          Soothsayer.new(base_config(%{regressors: ["temperature"], epochs: 1})),
          training
        )

      assert_raise ArgumentError, ~r/fitted with regressors \["temperature"\]/, fn ->
        Soothsayer.predict(fitted_model, Series.from_list([~D[2022-04-01]]))
      end
    end

    test "fitting without the regressor column raises" do
      dates = Date.range(~D[2022-01-01], ~D[2022-01-31]) |> Enum.to_list()
      training = DataFrame.new(%{"ds" => dates, "y" => Enum.map(dates, &Date.day_of_year/1)})

      assert_raise ArgumentError, ~r/Regressor column "temperature" not found/, fn ->
        Soothsayer.fit(Soothsayer.new(base_config(%{regressors: ["temperature"]})), training)
      end
    end

    test "regressors are a list of names or a map of name to options" do
      from_list = Soothsayer.new(%{regressors: ["temperature"]}).config.regressors
      from_map = Soothsayer.new(%{regressors: %{"temperature" => %{}}}).config.regressors

      assert from_list == from_map
      assert from_list == %{"temperature" => %{mode: :additive, regularization: nil, layers: []}}

      assert_raise ArgumentError, ~r/regressors must be a list/, fn ->
        Soothsayer.new(%{regressors: "temperature"})
      end

      assert_raise ArgumentError, ~r/column name strings/, fn ->
        Soothsayer.new(%{regressors: [:temperature]})
      end

      assert_raise ArgumentError, ~r/unknown option :lag for regressor "temperature"/, fn ->
        Soothsayer.new(%{regressors: %{"temperature" => %{lag: 1}}})
      end
    end
  end
end
