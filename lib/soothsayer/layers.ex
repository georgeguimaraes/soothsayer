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

  Only the trend gets a bias (`use_bias: true`). It is the one intercept of
  the model, as in NeuralProphet, so it carries the level of the series and
  the other components stay zero-centered offsets. With a bias on every
  component the biases all receive the same gradient and share the level
  between them, which puts part of the level into the seasonalities.
  """
  @spec position_dense(Axon.t(), String.t(), keyword()) :: Axon.t()
  def position_dense(input, name, opts \\ []) do
    use_bias = Keyword.get(opts, :use_bias, false)

    input
    |> Axon.dense(1, activation: :linear, use_bias: use_bias, name: name)
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
