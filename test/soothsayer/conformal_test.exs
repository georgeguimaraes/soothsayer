defmodule Soothsayer.ConformalTest do
  use ExUnit.Case, async: true

  alias Soothsayer.Conformal

  doctest Soothsayer.Conformal

  describe "q_hat/2" do
    test "takes the ceil((n + 1)(1 - alpha))-th smallest score" do
      scores = Enum.shuffle(1..20)

      # rank ceil(21 * 0.9) = 19
      assert Conformal.q_hat(scores, 0.1) == 19
      # rank ceil(21 * 0.5) = 11
      assert Conformal.q_hat(scores, 0.5) == 11
    end

    test "raises when the rank runs past the number of scores" do
      # alpha 0.1 needs ceil(1 / 0.1 - 1) = 9 scores
      assert Conformal.q_hat(Enum.to_list(1..9), 0.1) == 9

      assert_raise ArgumentError, ~r/at least 9 usable rows/, fn ->
        Conformal.q_hat(Enum.to_list(1..8), 0.1)
      end
    end
  end

  describe "bounds/4" do
    test "naive bounds sit q_hat around yhat, rows past the last step take the last q_hat" do
      calibration = %{alpha: 0.1, method: :naive, q_hat: %{1 => 1.0, 2 => 2.0}, count: 20}
      steps = Nx.tensor([[1], [2], [5]])
      combined = Nx.tensor([[10.0], [10.0], [10.0]])

      {lower, upper} = Conformal.bounds(calibration, steps, combined, %{})

      assert Nx.to_flat_list(lower) == [9.0, 8.0, 8.0]
      assert Nx.to_flat_list(upper) == [11.0, 12.0, 12.0]
    end

    test "cqr bounds push the outermost quantiles out, each side by its own q_hat" do
      calibration = %{alpha: {0.05, 0.05}, method: :cqr, q_hat: %{1 => {0.5, 2.0}}, count: 20}
      steps = Nx.tensor([[1], [1]])
      combined = Nx.tensor([[10.0], [10.0]])
      quantiles = %{0.1 => Nx.tensor([[8.0], [9.0]]), 0.9 => Nx.tensor([[12.0], [11.0]])}

      {lower, upper} = Conformal.bounds(calibration, steps, combined, quantiles)

      assert Nx.to_flat_list(lower) == [7.5, 8.5]
      assert Nx.to_flat_list(upper) == [14.0, 13.0]
    end
  end

  describe "calibrate/3 validation" do
    test "rejects a pair of alphas with the naive method and cqr without quantiles" do
      model = Soothsayer.new(%{trend: %{changepoints: 0}, epochs: 1})
      dates = Date.range(~D[2023-01-01], ~D[2023-03-31]) |> Enum.to_list()

      frame =
        Explorer.DataFrame.new(%{"ds" => dates, "y" => Enum.map(dates, &Date.day_of_year/1)})

      fitted = Soothsayer.fit(model, frame)

      assert_raise ArgumentError, ~r/needs method: :cqr/, fn ->
        Conformal.calibrate(fitted, frame, alpha: {0.05, 0.05})
      end

      assert_raise ArgumentError, ~r/needs quantiles configured/, fn ->
        Conformal.calibrate(fitted, frame, method: :cqr)
      end

      assert_raise ArgumentError, ~r/alpha must be/, fn ->
        Conformal.calibrate(fitted, frame, alpha: 1.5)
      end

      assert_raise ArgumentError, ~r/has not been fitted/, fn ->
        Conformal.calibrate(model, frame)
      end
    end
  end
end
