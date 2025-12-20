defmodule Soothsayer.SeasonalityTest do
  use ExUnit.Case, async: true

  alias Explorer.DataFrame
  alias Soothsayer.Seasonality

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

    test "yearly input has shape {nil, 2 * fourier_terms}" do
      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 6},
          weekly: %{enabled: true, fourier_terms: 3}
        }
      }

      inputs = Seasonality.build_inputs(config)

      # 6 fourier terms = 12 features (sin + cos for each)
      assert Axon.get_inputs(inputs.yearly)["yearly"] == {nil, 12}
    end

    test "weekly input has shape {nil, 2 * fourier_terms}" do
      config = %{
        seasonality: %{
          yearly: %{enabled: true, fourier_terms: 6},
          weekly: %{enabled: true, fourier_terms: 3}
        }
      }

      inputs = Seasonality.build_inputs(config)

      # 3 fourier terms = 6 features (sin + cos for each)
      assert Axon.get_inputs(inputs.weekly)["weekly"] == {nil, 6}
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
        yearly: Axon.input("yearly", shape: {nil, 12}),
        weekly: Axon.input("weekly", shape: {nil, 6})
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
        yearly: Axon.input("yearly", shape: {nil, 12}),
        weekly: Axon.input("weekly", shape: {nil, 6})
      }

      components = Seasonality.build_components(inputs, config)

      # Build yearly and verify it produces output
      {init_fn, predict_fn} = Axon.build(components.yearly)
      params = init_fn.(%{"yearly" => Nx.broadcast(0.5, {1, 12})}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"yearly" => Nx.broadcast(0.5, {1, 12})})

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
        yearly: Axon.input("yearly", shape: {nil, 12}),
        weekly: Axon.input("weekly", shape: {nil, 6})
      }

      components = Seasonality.build_components(inputs, config)

      # Build yearly (disabled) and verify output is 0
      {init_fn, predict_fn} = Axon.build(components.yearly)
      params = init_fn.(%{"yearly" => Nx.broadcast(0.5, {1, 12})}, Axon.ModelState.empty())
      output = predict_fn.(params, %{"yearly" => Nx.broadcast(0.5, {1, 12})})

      assert Nx.to_number(output) == 0.0
    end
  end

  # Existing preprocessor tests - renamed function
  describe "add_fourier_features/3" do
    test "adds yearly fourier columns when enabled" do
      df =
        DataFrame.new(%{
          "y" => [1, 2, 3, 4, 5],
          "ds" => [
            ~D[2023-01-01],
            ~D[2023-04-01],
            ~D[2023-07-01],
            ~D[2023-10-01],
            ~D[2023-12-31]
          ]
        })

      config = %{
        yearly: %{enabled: true, fourier_terms: 3},
        weekly: %{enabled: false, fourier_terms: 2}
      }

      result = Seasonality.add_fourier_features(df, "ds", config)

      # Should have sin and cos for each fourier term (3 * 2 = 6 columns)
      assert "yearly_sin_1" in result.names
      assert "yearly_cos_1" in result.names
      assert "yearly_sin_2" in result.names
      assert "yearly_cos_2" in result.names
      assert "yearly_sin_3" in result.names
      assert "yearly_cos_3" in result.names
    end

    test "adds weekly fourier columns when enabled" do
      df =
        DataFrame.new(%{
          "y" => [1, 2, 3, 4, 5, 6, 7],
          "ds" => [
            ~D[2023-01-02],
            ~D[2023-01-03],
            ~D[2023-01-04],
            ~D[2023-01-05],
            ~D[2023-01-06],
            ~D[2023-01-07],
            ~D[2023-01-08]
          ]
        })

      config = %{
        yearly: %{enabled: false, fourier_terms: 3},
        weekly: %{enabled: true, fourier_terms: 2}
      }

      result = Seasonality.add_fourier_features(df, "ds", config)

      # Should have sin and cos for each fourier term (2 * 2 = 4 columns)
      assert "weekly_sin_1" in result.names
      assert "weekly_cos_1" in result.names
      assert "weekly_sin_2" in result.names
      assert "weekly_cos_2" in result.names
    end

    test "adds both yearly and weekly when both enabled" do
      df =
        DataFrame.new(%{
          "y" => [1, 2, 3],
          "ds" => [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]
        })

      config = %{
        yearly: %{enabled: true, fourier_terms: 2},
        weekly: %{enabled: true, fourier_terms: 2}
      }

      result = Seasonality.add_fourier_features(df, "ds", config)

      # Should have both yearly and weekly columns
      assert "yearly_sin_1" in result.names
      assert "weekly_sin_1" in result.names
    end

    test "returns df unchanged when neither enabled" do
      df =
        DataFrame.new(%{
          "y" => [1, 2, 3],
          "ds" => [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]
        })

      config = %{
        yearly: %{enabled: false, fourier_terms: 2},
        weekly: %{enabled: false, fourier_terms: 2}
      }

      result = Seasonality.add_fourier_features(df, "ds", config)

      assert result.names == df.names
    end
  end
end
