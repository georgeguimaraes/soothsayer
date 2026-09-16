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

  describe "one_cycle/3" do
    test "rises to the peak at 30%, falls back by 60%, then decays to a tenth of the start" do
      schedule = Trainer.one_cycle(0.1, 100)
      at = fn step -> schedule.(Nx.tensor(step)) |> Nx.to_number() end

      assert_in_delta at.(0), 0.01, 1.0e-4
      assert_in_delta at.(30), 0.1, 1.0e-4
      assert_in_delta at.(60), 0.01, 1.0e-4
      assert at.(15) > at.(0) and at.(15) < at.(30)
      assert at.(99) < 0.0015
    end
  end

  describe "suggest_learning_rate/2" do
    test "picks the rate where the smoothed loss falls fastest, ignoring the edges and blow-ups" do
      # flat, then a steady drop between points 40 and 60, then flat, then divergence
      losses =
        List.duplicate(2.0, 40) ++
          Enum.map(1..20, &(2.0 - &1 * 0.075)) ++
          List.duplicate(0.5, 20) ++ [:infinity, :nan, 50.0]

      learning_rates = Enum.map(0..(length(losses) - 1), &(&1 * 1.0))

      picked = Trainer.suggest_learning_rate(losses, learning_rates)
      assert picked >= 42.0 and picked <= 58.0, "picked #{picked}"
    end

    test "ignores the bounces of a run that has diverged" do
      # a steady drop between points 20 and 40, then a finite blow-up whose
      # fall from 3000 to 1000 would be the steepest gradient of the curve
      losses =
        List.duplicate(2.0, 20) ++
          Enum.map(1..20, &(2.0 - &1 * 0.075)) ++
          List.duplicate(0.5, 20) ++ [5.0, 600.0, 3000.0, 1000.0, 2500.0, 800.0]

      learning_rates = Enum.map(0..(length(losses) - 1), &(&1 * 1.0))

      picked = Trainer.suggest_learning_rate(losses, learning_rates)
      assert picked >= 22.0 and picked <= 38.0, "picked #{picked}"
    end
  end

  describe "auto_epochs/1" do
    test "gives small datasets many passes and large ones few, within 20..500" do
      assert Trainer.auto_epochs(130) == 220
      assert Trainer.auto_epochs(2615) == 80
      assert Trainer.auto_epochs(10) == 500
      assert Trainer.auto_epochs(1_000_000) == 20
    end
  end

  describe "fit/5 with unresolved options" do
    test "refuses :auto values, they must be resolved first" do
      network =
        Axon.input("x", shape: {nil, 1})
        |> Axon.dense(1)
        |> then(&Axon.container(%{combined: &1}))

      x = %{"x" => Nx.tensor([[1.0], [2.0]])}
      y = Nx.tensor([[1.0], [2.0]])

      assert_raise ArgumentError, ~r/Resolve :auto/, fn ->
        Trainer.fit(network, x, y, 2, %{learning_rate: :auto})
      end
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

  describe "regularization_terms/2 and weighted_l1_penalty/2" do
    test "one lambda per kernel column, from the component that owns the column" do
      params = %Axon.ModelState{
        data: %{
          "events_dense" => %{"kernel" => Nx.tensor([[1.0], [-2.0], [4.0]])},
          "regressors_dense" => %{"kernel" => Nx.tensor([[3.0], [-1.0]])},
          "trend_dense" => %{"kernel" => Nx.tensor([[2.0], [2.0]])},
          "yearly_dense" => %{"kernel" => Nx.tensor([[0.5], [0.5]])}
        }
      }

      config = %{
        events: %{
          "a" => %{steps_before: 1, steps_after: 0, regularization: 0.5},
          "b" => %{steps_before: 0, steps_after: 0}
        },
        regressors: %{"x" => %{mode: :additive, regularization: 2.0}, "z" => %{mode: :additive}},
        trend: %{regularization: 0.1},
        seasonality: %{regularization: nil}
      }

      terms = Trainer.regularization_terms(params, config)

      assert terms |> Enum.map(&elem(&1, 0)) |> Enum.sort() ==
               ["events_dense", "regressors_dense", "trend_dense"]

      weights = Map.new(terms)
      assert Nx.to_flat_list(weights["events_dense"]) == [0.5, 0.5, 0.0]
      assert Nx.to_flat_list(weights["regressors_dense"]) == [2.0, 0.0]
      assert Nx.to_flat_list(weights["trend_dense"]) == [0.10000000149011612, 0.10000000149011612]

      # 0.5 * (1 + 2) + 2 * 3 + 0.1 * (2 + 2)
      assert_in_delta Nx.to_number(Trainer.weighted_l1_penalty(params, weights)), 7.9, 1.0e-5
    end

    test "seasonality regularization covers every seasonal layer and nothing gives no terms" do
      params = %Axon.ModelState{
        data: %{
          "yearly_dense" => %{"kernel" => Nx.tensor([[1.0], [1.0]])},
          "weekly_dense" => %{"kernel" => Nx.tensor([[1.0]])},
          "trend_dense" => %{"kernel" => Nx.tensor([[1.0]])}
        }
      }

      terms = Trainer.regularization_terms(params, %{seasonality: %{regularization: 0.3}})
      assert terms |> Enum.map(&elem(&1, 0)) |> Enum.sort() == ["weekly_dense", "yearly_dense"]

      assert Trainer.regularization_terms(params, %{}) == []
      assert Trainer.regularization_terms(params, %{trend: %{regularization: 0}}) == []
      assert Nx.to_number(Trainer.weighted_l1_penalty(params, [])) == 0.0
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
