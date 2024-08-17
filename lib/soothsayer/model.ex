defmodule Soothsayer.Model do
  import Axon

  defstruct [:params, :state, :nn_model]

  def build(input_shape) do
    inputs = Axon.input("input", shape: input_shape)

    trend =
      inputs
      |> dense(32, activation: :relu)
      |> dense(16, activation: :relu)
      |> dense(1, name: "trend")

    seasonality =
      inputs
      |> dense(64, activation: :relu)
      |> dense(32, activation: :relu)
      |> dense(1, name: "seasonality")

    ar =
      inputs
      |> dense(32, activation: :relu)
      |> dense(16, activation: :relu)
      |> dense(1, name: "ar")

    Axon.add([trend, seasonality, ar])
    |> dense(1, name: "output")
  end
end
