defmodule Soothsayer.SeriesTest do
  use ExUnit.Case, async: true

  alias Explorer.DataFrame
  alias Soothsayer.Series

  defp panel do
    DataFrame.new(%{
      "ds" => [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]],
      "y" => [1.0, 2.0, 10.0, 20.0, 30.0],
      "id" => ["b", "b", "a", "a", "a"]
    })
  end

  describe "split/2" do
    test "one frame per id, ids sorted, rows in their original order" do
      assert [{"a", a}, {"b", b}] = Series.split(panel(), %{series: %{column: "id"}})
      assert DataFrame.n_rows(a) == 3
      assert Explorer.Series.to_list(b["y"]) == [1.0, 2.0]
    end

    test "a single series model gets the whole frame under nil" do
      frame = panel()
      assert [{nil, ^frame}] = Series.split(frame, %{series: %{column: nil}})
    end

    test "raises on a missing column, non-string ids and a one-row series" do
      assert_raise ArgumentError, ~r/Series column "id" not found/, fn ->
        Series.split(DataFrame.select(panel(), ["ds", "y"]), %{series: %{column: "id"}})
      end

      numeric = DataFrame.put(panel(), "id", Explorer.Series.from_list([1, 1, 2, 2, 2]))

      assert_raise ArgumentError, ~r/ids must be strings/, fn ->
        Series.split(numeric, %{series: %{column: "id"}})
      end

      short = DataFrame.put(panel(), "id", Explorer.Series.from_list(["b", "b", "a", "a", "c"]))

      assert_raise ArgumentError, ~r/Series "c" needs at least 2 rows/, fn ->
        Series.split(short, %{series: %{column: "id"}})
      end
    end
  end

  describe "put_inputs/5" do
    test "adds the one-hot of the series and its level, broadcast over the rows" do
      config = %{series: %{column: "id", ids: ["a", "b", "c"]}}
      entry = %{normalization: %{mean: Nx.tensor([4.0]), std: Nx.tensor([2.0])}}

      x = Series.put_inputs(%{"trend" => Nx.iota({2, 1, 1})}, config, "b", entry, 2)

      assert Nx.to_list(x["series"]) == [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
      assert Nx.to_list(x["series_level"]) == [[2.0], [2.0]]

      # a series the model never saw selects nothing
      unknown = Series.put_inputs(%{"trend" => Nx.iota({1, 1, 1})}, config, "z", entry, 1)
      assert Nx.to_list(unknown["series"]) == [[0.0, 0.0, 0.0]]
    end

    test "leaves a single series model's inputs alone" do
      x = %{"trend" => Nx.iota({2, 1, 1})}
      assert Series.put_inputs(x, %{series: %{column: nil}}, nil, %{}, 2) == x
    end
  end

  describe "validate_config!/1" do
    defp series_config(overrides) do
      defaults = %{
        column: "id",
        normalize: :local,
        trend: :global,
        seasonality: :global,
        local_regularization: nil,
        unknown: :error
      }

      %{series: Map.merge(defaults, overrides)}
    end

    test "accepts a string column and the two normalize modes, rejects the rest" do
      assert Series.validate_config!(series_config(%{normalize: :global})) == :ok

      assert_raise ArgumentError, ~r/series must be/, fn ->
        Series.validate_config!(series_config(%{column: :id}))
      end

      assert_raise ArgumentError, ~r/series must be/, fn ->
        Series.validate_config!(series_config(%{normalize: :each}))
      end

      assert_raise ArgumentError, ~r/needs a global trend and seasonality/, fn ->
        Series.validate_config!(series_config(%{unknown: :global, trend: :local}))
      end
    end
  end

  describe "rows_for/3" do
    test "rows of the id when the frame has the column, the whole frame when it doesn't" do
      with_ids = panel()
      without_ids = DataFrame.select(with_ids, ["ds", "y"])

      assert Series.rows_for(with_ids, "id", "b") |> DataFrame.n_rows() == 2
      assert Series.rows_for(without_ids, "id", "b") == without_ids
      assert Series.rows_for(with_ids, nil, nil) == with_ids
      assert Series.rows_for(nil, "id", "b") == nil
    end
  end
end
