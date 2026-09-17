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
    end

    test "leaves a single series model's inputs alone" do
      x = %{"trend" => Nx.iota({2, 1, 1})}
      assert Series.put_inputs(x, %{series: %{column: nil}}, nil, %{}, 2) == x
    end
  end

  describe "validate_config!/1" do
    test "accepts a string column and the two normalize modes, rejects the rest" do
      assert Series.validate_config!(%{series: %{column: "id", normalize: :global}}) == :ok

      assert_raise ArgumentError, ~r/series must be/, fn ->
        Series.validate_config!(%{series: %{column: :id, normalize: :local}})
      end

      assert_raise ArgumentError, ~r/series must be/, fn ->
        Series.validate_config!(%{series: %{column: "id", normalize: :each}})
      end
    end
  end
end
