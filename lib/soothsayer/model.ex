defmodule Soothsayer.Model do
  defstruct [:nn_model, :params]

  def build do
    Axon.input("input", shape: {nil, 1})
    |> Axon.dense(1, activation: :linear)
  end

  def new do
    nn_model = build()
    %__MODULE__{nn_model: nn_model, params: nil}
  end
end
