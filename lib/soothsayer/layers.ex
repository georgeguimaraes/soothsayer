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
  A `position_dense/3` with one kernel per series, picked by the `series`
  one-hot input: `{batch, positions, features}` and `{batch, n_series}` in,
  `{batch, positions}` out. The kernel is `{n_series, features, 1}` and the
  bias, when asked for, `{n_series, 1}`, so `params.data[name]["kernel"][i]`
  is series `i`'s kernel and looks like the shared one.

  Local trend and seasonality use it, see `Soothsayer.Series`. The layer
  keeps the `name` of the shared layer it replaces, so regularization and
  the weight readers find it.
  """
  @spec series_dense(Axon.t(), Axon.t(), pos_integer(), String.t(), keyword()) :: Axon.t()
  def series_dense(input, series, n_series, name, opts \\ []) do
    use_bias = Keyword.get(opts, :use_bias, false)

    # The kernel shape follows the input's feature axis. Axon's glorot reads
    # the leading series axis as part of the receptive field, so the scale
    # keeps each series' kernel starting like a plain dense kernel.
    kernel =
      Axon.param(
        "kernel",
        fn input_shape, _series_shape -> {n_series, elem(input_shape, 2), 1} end,
        initializer: Axon.Initializers.glorot_uniform(scale: n_series)
      )

    if use_bias do
      bias = Axon.param("bias", {n_series, 1}, initializer: :zeros)

      Axon.layer(&series_dense_op/5, [input, series, kernel, bias],
        name: name,
        op_name: :series_dense
      )
    else
      Axon.layer(&series_dense_op/4, [input, series, kernel], name: name, op_name: :series_dense)
    end
  end

  defp series_dense_op(input, series, kernel, _opts) do
    kernel_for_sample = Nx.dot(series, [1], kernel, [0])

    input
    |> Nx.dot([2], [0], kernel_for_sample, [1], [0])
    |> Nx.squeeze(axes: [-1])
  end

  defp series_dense_op(input, series, kernel, bias, opts) do
    Nx.add(series_dense_op(input, series, kernel, opts), Nx.dot(series, [1], bias, [0]))
  end

  @doc """
  A `position_dense/3` over a slice of the feature axis, for inputs whose
  columns belong to more than one layer. `nil` when the range is `nil`.
  """
  @spec position_dense_over(Axon.t(), Range.t() | nil, String.t()) :: Axon.t() | nil
  def position_dense_over(_input, nil, _name), do: nil

  def position_dense_over(input, range, name) do
    input
    |> Axon.nx(fn tensor -> tensor[[.., .., range]] end, name: name <> "_columns")
    |> position_dense(name)
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
