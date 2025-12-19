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
