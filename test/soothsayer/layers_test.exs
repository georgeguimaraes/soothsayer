defmodule Soothsayer.LayersTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Layers

  describe "series_dense/5" do
    defp build(use_bias) do
      input = Axon.input("trend", shape: {nil, 3, 4})
      series = Axon.input("series", shape: {nil, 2})

      Layers.series_dense(input, series, 2, "trend_dense", use_bias: use_bias)
      |> then(&Axon.container(%{combined: &1}))
      |> Axon.build()
    end

    test "each row uses the kernel and bias of its own series" do
      {init_fn, predict_fn} = build(true)

      x = %{
        "trend" => Nx.iota({4, 3, 4}, type: :f32),
        "series" => Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
      }

      params = init_fn.(x, Axon.ModelState.empty())
      kernel = params.data["trend_dense"]["kernel"]
      bias = params.data["trend_dense"]["bias"]

      assert Nx.shape(kernel) == {2, 4, 1}
      assert Nx.shape(bias) == {2, 1}

      output = predict_fn.(params, x).combined
      assert Nx.shape(output) == {4, 3}

      for {row, series} <- [{0, 0}, {1, 1}, {2, 1}, {3, 0}] do
        expected =
          x["trend"][row]
          |> Nx.dot([1], kernel[series], [0])
          |> Nx.squeeze()
          |> Nx.add(bias[series])

        assert Nx.all_close(output[row], expected) |> Nx.to_number() == 1
      end
    end

    test "the kernel starts on the scale of a plain dense kernel" do
      {init_fn, _predict_fn} = build(false)

      x = %{"trend" => Nx.iota({2, 3, 4}, type: :f32), "series" => Nx.eye(2)}
      params = init_fn.(x, Axon.ModelState.empty())
      kernel = params.data["trend_dense"]["kernel"]

      # glorot uniform over fan_in 4 and fan_out 1 has bound sqrt(6 / 5)
      assert Nx.to_number(Nx.reduce_max(Nx.abs(kernel))) < :math.sqrt(6 / 5) + 1.0e-6
      assert Nx.to_number(Nx.reduce_max(Nx.abs(kernel))) > 0.3
      refute Map.has_key?(params.data["trend_dense"], "bias")
    end
  end
end
