defmodule Soothsayer.TrainerTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Trainer

  describe "fit/5" do
    test "trains network without regularization" do
      # Simple network: one dense layer
      network =
        Axon.input("x", shape: {nil, 2})
        |> Axon.dense(1, activation: :linear)
        |> then(&Axon.container(%{combined: &1}))

      x = %{"x" => Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
      y = Nx.tensor([[3.0], [7.0], [11.0]])

      config = %{learning_rate: 0.1}

      params = Trainer.fit(network, x, y, 10, config)

      assert params != nil
      assert is_struct(params, Axon.ModelState)
    end

    test "trains network with AR regularization" do
      network =
        Axon.input("x", shape: {nil, 2})
        |> Axon.dense(1, activation: :linear, name: "ar_dense_out")
        |> then(&Axon.container(%{combined: &1}))

      x = %{"x" => Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
      y = Nx.tensor([[3.0], [7.0], [11.0]])

      config = %{learning_rate: 0.1, ar: %{regularization: 0.1}}

      params = Trainer.fit(network, x, y, 10, config)

      assert params != nil
      assert is_struct(params, Axon.ModelState)
    end

    test "trains network with trend regularization" do
      network =
        Axon.input("x", shape: {nil, 2})
        |> Axon.dense(1, activation: :linear, name: "trend_dense")
        |> then(&Axon.container(%{combined: &1}))

      x = %{"x" => Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
      y = Nx.tensor([[3.0], [7.0], [11.0]])

      config = %{learning_rate: 0.1, trend: %{regularization: 0.1}}

      params = Trainer.fit(network, x, y, 10, config)

      assert params != nil
      assert is_struct(params, Axon.ModelState)
    end

    test "trains network with both AR and trend regularization" do
      ar_input = Axon.input("ar", shape: {nil, 2})
      trend_input = Axon.input("trend", shape: {nil, 2})

      ar = Axon.dense(ar_input, 1, activation: :linear, name: "ar_dense_out")
      trend = Axon.dense(trend_input, 1, activation: :linear, name: "trend_dense")
      combined = Axon.add([ar, trend])

      network = Axon.container(%{combined: combined, ar: ar, trend: trend})

      x = %{
        "ar" => Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "trend" => Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      }

      y = Nx.tensor([[3.0], [7.0], [11.0]])

      config = %{
        learning_rate: 0.1,
        ar: %{regularization: 0.1},
        trend: %{regularization: 0.1}
      }

      params = Trainer.fit(network, x, y, 10, config)

      assert params != nil
      assert is_struct(params, Axon.ModelState)
    end
  end

  describe "seed" do
    test "same seed gives identical parameters, no seed gives different ones" do
      network =
        Axon.input("x", shape: {nil, 2})
        |> Axon.dense(1, activation: :linear)
        |> then(&Axon.container(%{combined: &1}))

      x = %{"x" => Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])}
      y = Nx.tensor([[3.0], [7.0], [11.0], [15.0]])

      kernel = fn config ->
        Trainer.fit(network, x, y, 2, config).data["dense_0"]["kernel"] |> Nx.to_flat_list()
      end

      assert kernel.(%{learning_rate: 0.1, seed: 7}) == kernel.(%{learning_rate: 0.1, seed: 7})
      refute kernel.(%{learning_rate: 0.1, seed: 7}) == kernel.(%{learning_rate: 0.1, seed: 8})
    end
  end

  describe "batches/3" do
    test "shuffles rows into fixed-size batches and drops the leftover" do
      x = %{"a" => Nx.iota({100, 2}), "b" => Nx.iota({100, 3})}
      y = Nx.iota({100, 1})

      batches = Trainer.batches(x, y, 32)

      assert length(batches) == 3

      for {x_batch, y_batch} <- batches do
        assert Nx.shape(y_batch) == {32, 1}
        assert Nx.shape(x_batch["a"]) == {32, 2}
        assert Nx.shape(x_batch["b"]) == {32, 3}
      end

      # Rows stay aligned across inputs and target after shuffling:
      # a[i] == [2i, 2i + 1] and y[i] == [i] for every row of the iota
      for {x_batch, y_batch} <- batches do
        expected_a =
          Nx.concatenate([Nx.multiply(y_batch, 2), Nx.add(Nx.multiply(y_batch, 2), 1)], axis: 1)

        assert Nx.equal(x_batch["a"], expected_a) |> Nx.all() |> Nx.to_number() == 1
      end

      # Not sorted, so the shuffle happened
      seen = batches |> Enum.flat_map(fn {_, yb} -> Nx.to_flat_list(yb) end)
      assert seen != Enum.sort(seen)
      assert length(Enum.uniq(seen)) == 96
    end

    test "batch size equal to the row count yields a single batch" do
      x = %{"a" => Nx.iota({5, 1})}
      y = Nx.iota({5, 1})

      assert [{_x_batch, y_batch}] = Trainer.batches(x, y, 5)
      assert Nx.shape(y_batch) == {5, 1}
    end
  end

  describe "auto_batch_size/1" do
    test "grows with the order of magnitude of the data and clamps at both ends" do
      assert Trainer.auto_batch_size(5) == 16
      assert Trainer.auto_batch_size(100) == 16
      assert Trainer.auto_batch_size(1826) == 32
      assert Trainer.auto_batch_size(100_000) == 128
      assert Trainer.auto_batch_size(100_000_000) == 512
    end
  end

  describe "compute_l1_penalty/2" do
    test "returns zero for empty layer list" do
      params = %Axon.ModelState{data: %{}}
      result = Trainer.compute_l1_penalty(params, [])

      assert Nx.to_number(result) == 0.0
    end

    test "computes L1 penalty for single layer" do
      params = %Axon.ModelState{
        data: %{
          "dense" => %{"kernel" => Nx.tensor([[1.0], [-2.0], [3.0]])}
        }
      }

      result = Trainer.compute_l1_penalty(params, ["dense"])

      # |1| + |-2| + |3| = 6
      assert Nx.to_number(result) == 6.0
    end

    test "computes L1 penalty for multiple layers" do
      params = %Axon.ModelState{
        data: %{
          "layer1" => %{"kernel" => Nx.tensor([[1.0], [-1.0]])},
          "layer2" => %{"kernel" => Nx.tensor([[2.0], [-2.0]])}
        }
      }

      result = Trainer.compute_l1_penalty(params, ["layer1", "layer2"])

      # (|1| + |-1|) + (|2| + |-2|) = 2 + 4 = 6
      assert Nx.to_number(result) == 6.0
    end
  end
end
