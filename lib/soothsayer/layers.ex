defmodule Soothsayer.Layers do
  @moduledoc """
  Axon layers shared by the components.
  """

  @doc """
  A linear layer with one output per position: `{batch, positions, features}`
  in, `{batch, positions}` out, with one kernel shared across the positions.

  Every component that depends only on the timestamp uses it, so the same
  coefficients apply at the lag positions, where the component is subtracted
  from the lags, and at the target positions, where it is part of the
  forecast. The dense layer is named `name` so its weights can be found for
  regularization and inspection.
  """
  @spec position_dense(Axon.t(), String.t()) :: Axon.t()
  def position_dense(input, name) do
    input
    |> Axon.dense(1, activation: :linear, name: name)
    |> Axon.nx(&Nx.squeeze(&1, axes: [-1]), name: name <> "_positions")
  end

  @doc """
  Slices the positions axis of a component output. Disabled components are
  scalar constants and pass through untouched, since they broadcast anyway.
  """
  @spec slice_positions(Axon.t(), Range.t(), String.t()) :: Axon.t()
  def slice_positions(component, range, name) do
    Axon.nx(
      component,
      fn tensor ->
        if Nx.rank(tensor) == 0, do: tensor, else: tensor[[.., range]]
      end,
      name: name
    )
  end
end
