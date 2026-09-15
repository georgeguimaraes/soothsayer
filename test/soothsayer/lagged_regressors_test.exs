defmodule Soothsayer.LaggedRegressorsTest do
  use ExUnit.Case, async: true

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.LaggedRegressors

  describe "origins with unequal lags" do
    test "the longest lag window decides the first origin and every input has the same rows" do
      y = Nx.iota({10}, type: :f32)
      regressor = Nx.multiply(Nx.iota({10}, type: :f32), 10)

      rows = AR.training_rows(y, 2, 1, max_lags: 4)
      regressor_rows = AR.lagged_rows(regressor, rows.origin_indices, 4, 1)

      assert rows.origin_indices == [3, 4, 5, 6, 7, 8]
      assert Nx.shape(rows.lagged) == {6, 2}
      assert Nx.shape(regressor_rows) == {6, 4}
      assert Nx.to_list(rows.lagged) |> hd() == [2.0, 3.0]
      assert Nx.to_list(regressor_rows) |> hd() == [0.0, 10.0, 20.0, 30.0]
      assert Nx.to_flat_list(rows.targets) |> hd() == 4.0
    end
  end

  describe "build_input/3" do
    test "reads each regressor's window ending at the origin and raises on a missing date" do
      config = %{lagged_regressors: %{"temperature" => %{lags: 2}, "price" => %{lags: 1}}}

      known_values = %{
        "temperature" => %{~D[2023-01-01] => 10.0, ~D[2023-01-02] => 11.0, ~D[2023-01-03] => 12.0},
        "price" => %{~D[2023-01-02] => 5.0, ~D[2023-01-03] => 6.0}
      }

      result =
        LaggedRegressors.build_input(known_values, [~D[2023-01-02], ~D[2023-01-03]], config)

      # columns: price (1 lag) then temperature (2 lags), sorted by name
      assert Nx.to_list(result) == [[5.0, 10.0, 11.0], [6.0, 11.0, 12.0]]

      assert_raise ArgumentError, ~r/"price" has no value for 2023-01-01/, fn ->
        LaggedRegressors.build_input(known_values, [~D[2023-01-01]], config)
      end
    end
  end

  describe "config" do
    test "requires auto-regression and a well formed map" do
      assert_raise ArgumentError, ~r/need auto-regression enabled/, fn ->
        Soothsayer.new(%{lagged_regressors: %{"temperature" => %{lags: 2}}})
      end

      assert_raise ArgumentError, ~r/must map column names/, fn ->
        Soothsayer.new(%{ar: %{enabled: true, lags: 2}, lagged_regressors: %{"temperature" => 2}})
      end
    end
  end

  describe "fitting and predicting" do
    # y depends on yesterday's temperature, which the dates and y's own past can't explain
    defp lagged_data(dates) do
      temperature = Enum.map(dates, fn _ -> 20 + :rand.normal(0, 4) end)
      lagged_temperature = [hd(temperature) | temperature] |> Enum.take(length(dates))

      y = Enum.map(lagged_temperature, fn temp -> 100 + 2 * temp + :rand.normal(0, 1) end)
      DataFrame.new(%{"ds" => dates, "y" => y, "temperature" => temperature})
    end

    defp lagged_model(overrides) do
      Soothsayer.new(
        Map.merge(
          %{
            trend: %{enabled: false, changepoints: 0},
            seasonality: %{yearly: %{enabled: false}, weekly: %{enabled: false}},
            ar: %{enabled: true, lags: 2},
            epochs: 40,
            seed: 5
          },
          overrides
        )
      )
    end

    test "yesterday's regressor cuts one-step error and shows up as a component" do
      :rand.seed(:exsss, {3, 1, 4})
      dates = Date.range(~D[2021-01-01], ~D[2022-12-31]) |> Enum.to_list()
      holdout_dates = Date.range(~D[2023-01-01], ~D[2023-03-31]) |> Enum.to_list()
      training = lagged_data(dates)
      holdout = lagged_data(holdout_dates)

      one_step_error = fn fitted_model ->
        components =
          Soothsayer.predict_components(fitted_model, holdout["ds"],
            history: holdout,
            regressors: holdout
          )

        error =
          components.combined
          |> Nx.flatten()
          |> Nx.subtract(holdout["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32}))
          |> Nx.abs()
          |> Nx.mean()
          |> Nx.to_number()

        {error, components}
      end

      {without_error, _} = one_step_error.(Soothsayer.fit(lagged_model(%{}), training))

      with_model =
        Soothsayer.fit(
          lagged_model(%{lagged_regressors: %{"temperature" => %{lags: 2}}}),
          training
        )

      {with_error, components} = one_step_error.(with_model)

      assert with_error < without_error * 0.5
      assert components.lagged_regressors |> Nx.abs() |> Nx.sum() |> Nx.to_number() > 0

      summed =
        [
          :trend,
          :yearly_seasonality,
          :weekly_seasonality,
          :ar,
          :events,
          :regressors,
          :lagged_regressors
        ]
        |> Enum.map(&components[&1])
        |> Enum.reduce(&Nx.add/2)

      assert Nx.all_close(summed, components.combined, atol: 1.0e-2) |> Nx.to_number() == 1
    end

    test "forecasting past the training data only needs regressor values up to each block origin" do
      :rand.seed(:exsss, {3, 1, 4})
      dates = Date.range(~D[2022-01-01], ~D[2022-06-30]) |> Enum.to_list()
      training = lagged_data(dates)

      fitted_model =
        Soothsayer.fit(
          lagged_model(%{
            lagged_regressors: %{"temperature" => %{lags: 2}},
            ar: %{enabled: true, lags: 2, forecast_steps: 2},
            epochs: 2
          }),
          training
        )

      last_date = List.last(dates)

      # First block (2 days) needs temperature through the last training date only
      first_block =
        Soothsayer.predict(
          fitted_model,
          Series.from_list([Date.add(last_date, 1), Date.add(last_date, 2)])
        )

      assert Nx.shape(first_block) == {2, 1}

      # The second block origin is last_date + 2; its window starts at last_date + 1, which has no temperature yet
      assert_raise ArgumentError, ~r/no value for 2022-07-01/, fn ->
        Soothsayer.predict(fitted_model, Series.from_list([Date.add(last_date, 3)]))
      end

      newer =
        DataFrame.new(%{
          "ds" => [Date.add(last_date, 1), Date.add(last_date, 2)],
          "temperature" => [21.0, 19.0]
        })

      third_day =
        Soothsayer.predict(fitted_model, Series.from_list([Date.add(last_date, 3)]),
          regressors: newer
        )

      assert Nx.shape(third_day) == {1, 1}
    end
  end
end
