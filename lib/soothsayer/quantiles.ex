defmodule Soothsayer.Quantiles do
  @moduledoc """
  Uncertainty estimation through quantile regression.

  With `quantiles: [0.1, 0.9]` the network grows one linear head per
  quantile on top of the same inputs the components use (trend features,
  Fourier terms, lags, step mask, events and regressors). Each head learns
  how far that quantile sits from the median forecast, and is trained with
  the pinball loss for its quantile while the median keeps training on the
  Huber loss. The median is detached before the heads are added to it, so
  the quantile losses don't pull it around.

  Heads for quantiles above 0.5 are added to the median and heads below are
  subtracted, and `Soothsayer.predict_components/3` clips them so an upper
  quantile never falls below the median and a lower one never rises above
  it, which is what NeuralProphet does at predict time too.
  """

  @doc """
  Builds one quantile head per configured quantile.

  ## Parameters

    * `inputs` - Every Axon input node in the network, concatenated as the
      head's features.
    * `combined` - The median forecast node.
    * `config` - Model configuration with a sorted `:quantiles` list.

  ## Returns

    A list of Axon nodes in the same order as `config.quantiles`, empty when
    no quantiles are configured.

  """
  @spec build_components(list(Axon.t()), Axon.t(), map()) :: list(Axon.t())
  def build_components(_inputs, _combined, %{quantiles: []}), do: []

  def build_components(inputs, combined, %{quantiles: quantiles}) do
    features = concatenate(inputs)
    median = Axon.nx(combined, &Nx.Defn.Kernel.stop_grad/1, name: "median_detached")

    Enum.map(quantiles, fn quantile ->
      deviation = Axon.dense(features, 1, activation: :linear, name: layer_name(quantile))

      if quantile > 0.5 do
        Axon.add(median, deviation)
      else
        Axon.subtract(median, deviation)
      end
    end)
  end

  def build_components(_inputs, _combined, _config), do: []

  defp concatenate([single]), do: single
  defp concatenate(inputs), do: Axon.concatenate(inputs, axis: 1)

  @doc """
  Name of the dense layer for a quantile, e.g. `"quantile_0_900_dense"` for 0.9.
  """
  @spec layer_name(float()) :: String.t()
  def layer_name(quantile) do
    digits = quantile |> :erlang.float_to_binary(decimals: 3) |> String.replace(".", "_")
    "quantile_#{digits}_dense"
  end

  @doc """
  Pinball (quantile) loss, averaged over all elements.

  For error `d = target - prediction` the loss is `max(q * d, (q - 1) * d)`,
  which is minimized when `prediction` is the q-th quantile of the target.

  ## Examples

      iex> Soothsayer.Quantiles.pinball_loss(Nx.tensor([[1.0], [3.0]]), Nx.tensor([[2.0], [2.0]]), 0.9) |> Nx.to_number()
      0.5

  """
  @spec pinball_loss(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def pinball_loss(targets, predictions, quantile) do
    error = Nx.subtract(targets, predictions)

    Nx.max(Nx.multiply(quantile, error), Nx.multiply(quantile - 1, error))
    |> Nx.mean()
  end

  @doc """
  Validates a `quantiles` config value: a list of numbers strictly between 0
  and 1. Returns the quantiles sorted, as floats, without duplicates.
  """
  @spec normalize_config!(term()) :: list(float())
  def normalize_config!(quantiles) when is_list(quantiles) do
    for quantile <- quantiles, not (is_number(quantile) and quantile > 0 and quantile < 1) do
      raise ArgumentError,
            "quantiles must be numbers strictly between 0 and 1, got #{inspect(quantile)}"
    end

    quantiles |> Enum.map(&(&1 * 1.0)) |> Enum.uniq() |> Enum.sort()
  end

  def normalize_config!(quantiles) do
    raise ArgumentError, "quantiles must be a list, got #{inspect(quantiles)}"
  end
end
