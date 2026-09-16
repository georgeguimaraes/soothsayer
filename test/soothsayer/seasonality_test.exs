defmodule Soothsayer.SeasonalityTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Seasonality

  doctest Seasonality

  describe "build_inputs/1" do
    test "returns map with yearly and weekly inputs" do
      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 6},
          weekly: %{enabled: true, fourier_terms: 3}
        }
      }

      inputs = Seasonality.build_inputs(config)

      assert Map.has_key?(inputs, :yearly)
      assert Map.has_key?(inputs, :weekly)
    end

    test "yearly input has shape {nil, positions, 2 * fourier_terms}" do
      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 6},
          weekly: %{enabled: true, fourier_terms: 3}
        }
      }

      inputs = Seasonality.build_inputs(config)

      # 6 fourier terms = 12 features (sin + cos for each)
      assert Axon.get_inputs(inputs.yearly)["yearly"] == {nil, 1, 12}
    end

    test "weekly input has shape {nil, positions, 2 * fourier_terms}" do
      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 6},
          weekly: %{enabled: true, fourier_terms: 3}
        }
      }

      inputs = Seasonality.build_inputs(config)

      # 3 fourier terms = 6 features (sin + cos for each)
      assert Axon.get_inputs(inputs.weekly)["weekly"] == {nil, 1, 6}
    end
  end

  describe "build_components/2" do
    test "returns map with yearly and weekly components" do
      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 6},
          weekly: %{enabled: true, fourier_terms: 3}
        }
      }

      inputs = %{
        yearly: Axon.input("yearly", shape: {nil, 1, 12}),
        weekly: Axon.input("weekly", shape: {nil, 1, 6})
      }

      components = Seasonality.build_components(inputs, config)

      assert Map.has_key?(components, :yearly)
      assert Map.has_key?(components, :weekly)
    end

    test "returns dense layer when period is enabled" do
      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 6},
          weekly: %{enabled: false, fourier_terms: 3}
        }
      }

      inputs = %{
        yearly: Axon.input("yearly", shape: {nil, 1, 12}),
        weekly: Axon.input("weekly", shape: {nil, 1, 6})
      }

      components = Seasonality.build_components(inputs, config)

      # Build yearly and verify it produces output
      {init_fn, predict_fn} = Axon.build(components.yearly)
      params = init_fn.(%{"yearly" => Nx.broadcast(0.5, {1, 1, 12})}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"yearly" => Nx.broadcast(0.5, {1, 1, 12})})

      # Output should be non-zero (dense layer learned weights)
      assert Nx.shape(output) == {1, 1}
    end

    test "returns constant 0 when period is disabled" do
      config = %{
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 6},
          weekly: %{enabled: true, fourier_terms: 3}
        }
      }

      inputs = %{
        yearly: Axon.input("yearly", shape: {nil, 1, 12}),
        weekly: Axon.input("weekly", shape: {nil, 1, 6})
      }

      components = Seasonality.build_components(inputs, config)

      # Build yearly (disabled) and verify output is 0
      {init_fn, predict_fn} = Axon.build(components.yearly)
      params = init_fn.(%{"yearly" => Nx.broadcast(0.5, {1, 1, 12})}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"yearly" => Nx.broadcast(0.5, {1, 1, 12})})

      assert Nx.to_number(output) == 0.0
    end
  end

  # Existing preprocessor tests - renamed function
  describe "build_features/2" do
    test "returns map with yearly and weekly tensors" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]

      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 3},
          weekly: %{enabled: true, fourier_terms: 2}
        }
      }

      result = Seasonality.build_features(dates, config)

      assert is_map(result)
      assert Map.has_key?(result, :yearly)
      assert Map.has_key?(result, :weekly)
    end

    test "yearly tensor has shape {n_dates, 2 * fourier_terms}" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04], ~D[2023-01-05]]

      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 3},
          weekly: %{enabled: true, fourier_terms: 2}
        }
      }

      result = Seasonality.build_features(dates, config)

      # 3 fourier terms = 6 features (sin + cos for each)
      assert Nx.shape(result.yearly) == {5, 6}
    end

    test "weekly tensor has shape {n_dates, 2 * fourier_terms}" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03], ~D[2023-01-04], ~D[2023-01-05]]

      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 3},
          weekly: %{enabled: true, fourier_terms: 2}
        }
      }

      result = Seasonality.build_features(dates, config)

      # 2 fourier terms = 4 features (sin + cos for each)
      assert Nx.shape(result.weekly) == {5, 4}
    end

    test "returns zero tensors when seasonality is disabled" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]

      config = %{
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 3},
          weekly: %{enabled: false, fourier_terms: 2}
        }
      }

      result = Seasonality.build_features(dates, config)

      # Should still have correct shapes, but all zeros
      assert Nx.shape(result.yearly) == {3, 6}
      assert Nx.shape(result.weekly) == {3, 4}
      assert Nx.to_flat_list(result.yearly) |> Enum.all?(&(&1 == 0.0))
      assert Nx.to_flat_list(result.weekly) |> Enum.all?(&(&1 == 0.0))
    end

    test "columns are sin and cos of each term in order" do
      dates = [~D[2023-01-01], ~D[2023-03-15]]
      config = %{seasonality: %{yearly: %{enabled: true, fourier_terms: 3}}}
      %{yearly: features} = Seasonality.build_features(dates, config)

      expected =
        dates
        |> Seasonality.compute_period_fractions(:yearly)
        |> Enum.map(fn fraction ->
          Enum.flat_map(1..3, fn term ->
            angle = 2 * :math.pi() * term * fraction
            [:math.sin(angle), :math.cos(angle)]
          end)
        end)

      assert Nx.all_close(features, Nx.tensor(expected), atol: 1.0e-6) |> Nx.to_number() == 1
    end

    test "tensors are f32 type" do
      dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]

      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 2},
          weekly: %{enabled: true, fourier_terms: 2}
        }
      }

      result = Seasonality.build_features(dates, config)

      assert Nx.type(result.yearly) == {:f, 32}
      assert Nx.type(result.weekly) == {:f, 32}
    end
  end

  describe "sub-daily timestamps" do
    test "yearly and weekly fractions at midnight equal the plain date's" do
      dates = [~D[2023-03-15], ~D[2024-02-29], ~D[2023-12-31]]
      midnights = Enum.map(dates, &NaiveDateTime.new!(&1, ~T[00:00:00]))

      for period <- [:yearly, :weekly] do
        assert Seasonality.compute_period_fractions(dates, period) ==
                 Seasonality.compute_period_fractions(midnights, period)
      end
    end

    test "the time of day moves the fractions within the day" do
      noon = [~N[2023-03-15 12:00:00]]

      assert Seasonality.compute_period_fractions(noon, :daily) == [0.5]
      assert Seasonality.compute_period_fractions(noon, :weekly) == [(3 + 0.5) / 7]
      assert Seasonality.compute_period_fractions(noon, :yearly) == [(74 + 0.5) / 365]
    end

    test "daily features vary with the hour and are zeros when disabled" do
      timestamps = Enum.map(0..5, &NaiveDateTime.add(~N[2023-01-01 00:00:00], &1 * 4, :hour))
      config = %{seasonality: %{daily: %{enabled: true, fourier_terms: 2}}}

      result = Seasonality.build_features(timestamps, config)

      assert Map.keys(result) == [:daily]
      assert Nx.shape(result.daily) == {6, 4}
      assert Nx.to_flat_list(result.daily) |> Enum.uniq() |> length() > 1

      disabled =
        Seasonality.build_features(
          timestamps,
          put_in(config, [:seasonality, :daily, :enabled], false)
        )

      assert Nx.to_flat_list(disabled.daily) |> Enum.all?(&(&1 == 0.0))
    end
  end

  describe "resolve_auto/3" do
    test "three years of daily data turn daily off and yearly on" do
      timestamps = [~D[2020-01-01], ~D[2023-01-01]]
      config = %{yearly: %{enabled: :auto}, weekly: %{enabled: :auto}, daily: %{enabled: :auto}}

      resolved = Seasonality.resolve_auto(config, timestamps, {1, :day})

      assert resolved == %{
               yearly: %{enabled: true},
               weekly: %{enabled: true},
               daily: %{enabled: false}
             }
    end

    test "explicit true and false are left alone" do
      config = %{yearly: %{enabled: true}, weekly: %{enabled: false}, daily: %{enabled: false}}

      assert Seasonality.resolve_auto(config, [~D[2023-01-01], ~D[2023-01-02]], {1, :day}) ==
               config
    end
  end

  describe "custom and conditional seasonalities" do
    test "custom periods are listed after the built-in ones and are always enabled" do
      config = %{
        seasonality: %{
          yearly: %{enabled: false, fourier_terms: 6},
          custom: %{
            "monthly" => %{period: 30.5, fourier_terms: 2},
            "biweekly" => %{period: 14, fourier_terms: 1}
          }
        }
      }

      assert Seasonality.periods(config) == [:yearly, :weekly, :daily, :biweekly, :monthly]
      assert Seasonality.enabled?(config, :monthly)
      refute Seasonality.enabled?(config, :yearly)

      features = Seasonality.build_features([~D[2023-01-01], ~D[2023-03-03]], config)
      assert Nx.shape(features.monthly) == {2, 4}
      assert Nx.shape(features.biweekly) == {2, 2}
      # 61 days apart is two full 30.5 day periods, so the same phase
      assert Nx.all_close(features.monthly[0], features.monthly[1], atol: 1.0e-4)
             |> Nx.to_number() == 1
    end

    test "a condition scales the period's features row by row and 0 outside the required set" do
      config = %{seasonality: %{weekly: %{enabled: true, fourier_terms: 1, condition: "summer"}}}
      dates = [~D[2023-01-02], ~D[2023-07-03], ~D[2023-07-04]]
      naive = Enum.map(dates, &NaiveDateTime.new!(&1, ~T[00:00:00]))

      conditions = %{"summer" => %{Enum.at(naive, 0) => 0.0, Enum.at(naive, 1) => 1.0}}

      plain =
        Seasonality.build_features(dates, %{
          seasonality: %{weekly: %{enabled: true, fourier_terms: 1}}
        })

      masked =
        Seasonality.build_features(dates, config, conditions,
          required: MapSet.new(Enum.take(naive, 2))
        )

      assert Nx.to_flat_list(masked.weekly[0]) == [0.0, 0.0]
      assert Nx.to_flat_list(masked.weekly[1]) == Nx.to_flat_list(plain.weekly[1])
      assert Nx.to_flat_list(masked.weekly[2]) == [0.0, 0.0]

      assert_raise ArgumentError, ~r/condition "summer" has no value for 2023-07-04/, fn ->
        Seasonality.build_features(dates, config, conditions)
      end

      assert Seasonality.condition_columns(config) == ["summer"]
    end

    test "condition values come from the dataframe as 0 to 1 floats" do
      config = %{seasonality: %{weekly: %{enabled: true, fourier_terms: 1, condition: "summer"}}}

      frame =
        Explorer.DataFrame.new(%{
          "ds" => [~D[2023-01-02], ~D[2023-07-03]],
          "summer" => [false, true]
        })

      values = Seasonality.condition_values(frame, config)
      assert values["summer"][~N[2023-01-02 00:00:00]] == 0.0
      assert values["summer"][~N[2023-07-03 00:00:00]] == 1.0

      bad = Explorer.DataFrame.new(%{"ds" => [~D[2023-01-02]], "summer" => [2.0]})

      assert_raise ArgumentError, ~r/must be between 0 and 1 or boolean/, fn ->
        Seasonality.condition_values(bad, config)
      end

      assert_raise ArgumentError, ~r/condition column "summer" not found/, fn ->
        Seasonality.condition_values(Explorer.DataFrame.new(%{"ds" => [~D[2023-01-02]]}), config)
      end
    end

    test "rejects unknown keys, bad names and bad specs" do
      assert_raise ArgumentError, ~r/unknown seasonality key :monthly/, fn ->
        Soothsayer.new(%{seasonality: %{monthly: %{period: 30.5, fourier_terms: 3}}})
      end

      assert_raise ArgumentError, ~r/lowercase identifiers other than yearly/, fn ->
        Soothsayer.new(%{seasonality: %{custom: %{"weekly" => %{period: 7, fourier_terms: 3}}}})
      end

      assert_raise ArgumentError, ~r/lowercase identifiers/, fn ->
        Soothsayer.new(%{
          seasonality: %{custom: %{"Monthly cycle" => %{period: 30.5, fourier_terms: 3}}}
        })
      end

      assert_raise ArgumentError, ~r/custom.monthly.period must be a positive number/, fn ->
        Soothsayer.new(%{seasonality: %{custom: %{"monthly" => %{fourier_terms: 3}}}})
      end

      assert_raise ArgumentError,
                   ~r/custom.monthly.fourier_terms must be a positive integer/,
                   fn ->
                     Soothsayer.new(%{
                       seasonality: %{custom: %{"monthly" => %{period: 30.5, fourier_terms: 0}}}
                     })
                   end

      assert_raise ArgumentError, ~r/seasonality.weekly.condition must be a column name/, fn ->
        Soothsayer.new(%{seasonality: %{weekly: %{condition: :summer}}})
      end
    end
  end
end
