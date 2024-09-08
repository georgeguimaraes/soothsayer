defmodule Soothsayer.PreprocessorTest do
  use ExUnit.Case, async: true
  alias Soothsayer.Preprocessor
  alias Explorer.DataFrame
  alias Explorer.Series

  describe "prepare_data/4" do
    test "prepares data with yearly and weekly seasonality" do
      df = DataFrame.new(%{
        "y" => [1, 2, 3, 4, 5],
        "ds" => [
          ~D[2023-01-01],
          ~D[2023-04-01],
          ~D[2023-07-01],
          ~D[2023-10-01],
          ~D[2024-01-01]
        ]
      })

      seasonality_config = %{
        yearly: %{enabled: true, fourier_terms: 3},
        weekly: %{enabled: true, fourier_terms: 2}
      }

      result = Preprocessor.prepare_data(df, "y", "ds", seasonality_config)

      assert result["y"] == df["y"]
      assert result["ds"] == df["ds"]
      assert "yearly_sin_1" in DataFrame.names(result)
      assert "yearly_cos_1" in DataFrame.names(result)
      assert "yearly_sin_3" in DataFrame.names(result)
      assert "yearly_cos_3" in DataFrame.names(result)
      assert "weekly_sin_1" in DataFrame.names(result)
      assert "weekly_cos_1" in DataFrame.names(result)
      assert "weekly_sin_2" in DataFrame.names(result)
      assert "weekly_cos_2" in DataFrame.names(result)
    end

    test "prepares data with only yearly seasonality" do
      df = DataFrame.new(%{
        "y" => [1, 2, 3, 4, 5],
        "ds" => [
          ~D[2023-01-01],
          ~D[2023-04-01],
          ~D[2023-07-01],
          ~D[2023-10-01],
          ~D[2024-01-01]
        ]
      })

      seasonality_config = %{
        yearly: %{enabled: true, fourier_terms: 3},
        weekly: %{enabled: false, fourier_terms: 2}
      }

      result = Preprocessor.prepare_data(df, "y", "ds", seasonality_config)

      assert result["y"] == df["y"]
      assert result["ds"] == df["ds"]
      assert "yearly_sin_1" in DataFrame.names(result)
      assert "yearly_cos_1" in DataFrame.names(result)
      assert "yearly_sin_3" in DataFrame.names(result)
      assert "yearly_cos_3" in DataFrame.names(result)
      refute "weekly_sin_1" in DataFrame.names(result)
      refute "weekly_cos_1" in DataFrame.names(result)
    end

    test "prepares data with no seasonality" do
      df = DataFrame.new(%{
        "y" => [1, 2, 3, 4, 5],
        "ds" => [
          ~D[2023-01-01],
          ~D[2023-04-01],
          ~D[2023-07-01],
          ~D[2023-10-01],
          ~D[2024-01-01]
        ]
      })

      seasonality_config = %{
        yearly: %{enabled: false, fourier_terms: 3},
        weekly: %{enabled: false, fourier_terms: 2}
      }

      result = Preprocessor.prepare_data(df, "y", "ds", seasonality_config)

      assert result["y"] == df["y"]
      assert result["ds"] == df["ds"]
      assert DataFrame.names(result) == ["y", "ds"]
    end
  end
end
