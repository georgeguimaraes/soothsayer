defmodule Soothsayer.AR do
  @moduledoc """
  Auto-regression (AR) component functionality.

  Handles network building, feature engineering, and weight extraction for AR models.
  Supports both linear AR and deep AR-Net architectures with configurable hidden layers.
  """

  # Network Building

  @doc """
  Creates the Axon input node for the AR component.

  ## Parameters

    * `config` - Model configuration map with `:ar` key.

  ## Returns

    An Axon input node when AR is enabled, `nil` otherwise.

  """
  @spec build_network_input(map()) :: Axon.t() | nil
  def build_network_input(%{ar: %{enabled: true, lags: lags}}) do
    Axon.input("ar", shape: {nil, lags})
  end
  def build_network_input(_config), do: nil

  @doc """
  Builds the AR component layer(s).

  Supports both linear AR (single dense layer) and deep AR-Net (multiple hidden layers
  with ReLU activation followed by linear output).

  ## Parameters

    * `input` - Axon input node from `build_network_input/1`.
    * `config` - Model configuration map.

  ## Returns

    An Axon layer when AR is enabled, `Axon.constant(0)` otherwise.

  """
  @spec build_component(Axon.t() | nil, map()) :: Axon.t()
  def build_component(nil, _config), do: Axon.constant(0)

  def build_component(input, %{ar: %{enabled: true} = ar_config}) do
    layers = Map.get(ar_config, :layers, [])
    build_ar_network(input, layers)
  end

  def build_component(_input, _config), do: Axon.constant(0)

  defp build_ar_network(input, []) do
    Axon.dense(input, 1, activation: :linear, name: "ar_dense_out")
  end

  defp build_ar_network(input, layers) do
    hidden = build_hidden_layers(input, layers)
    Axon.dense(hidden, 1, activation: :linear, name: "ar_dense_out")
  end

  defp build_hidden_layers(input, layers) do
    {hidden, _idx} =
      Enum.reduce(layers, {input, 0}, fn units, {acc, idx} ->
        {Axon.dense(acc, units, activation: :relu, name: "ar_dense_#{idx}"), idx + 1}
      end)

    hidden
  end

  # Feature Engineering

  @doc """
  Creates lagged input features and corresponding targets for AR training.

  Given a time series y and number of lags, creates sliding windows where each
  window contains lags consecutive values, and the target is the next value.

  ## Parameters

    * `y` - A 1D tensor of time series values.
    * `lags` - Number of lagged values to use as features.

  ## Returns

    A tuple `{lagged, targets}` where:
    * `lagged` - Tensor of shape `{n_samples, lags}` with lagged features
    * `targets` - Tensor of shape `{n_samples, 1}` with target values

  ## Examples

      iex> y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {lagged, targets} = Soothsayer.AR.create_lagged_inputs(y, 3)
      iex> Nx.shape(lagged)
      {2, 3}

  """
  @spec create_lagged_inputs(Nx.Tensor.t(), non_neg_integer()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def create_lagged_inputs(y, lags) do
    n = Nx.size(y)
    n_samples = n - lags

    # Build lagged features using Nx.slice for each lag position
    # lag 0: y[0:n_samples], lag 1: y[1:n_samples+1], etc.
    lagged =
      0..(lags - 1)
      |> Enum.map(fn lag -> Nx.slice(y, [lag], [n_samples]) end)
      |> Nx.stack(axis: 1)
      |> Nx.as_type({:f, 32})

    # Targets are the values after each window: y[lags:n]
    targets =
      y
      |> Nx.slice([lags], [n_samples])
      |> Nx.reshape({:auto, 1})
      |> Nx.as_type({:f, 32})

    {lagged, targets}
  end

  @doc """
  Builds AR input tensor for prediction given training data and prediction dates.

  For each prediction date, looks up the previous lags values from the training
  data to use as AR features. Returns zeros for dates that don't have enough history.

  ## Parameters

    * `training_data` - Map with `:dates` (list of dates) and `:y_normalized` (list of values)
    * `prediction_dates` - List of dates to build AR inputs for
    * `lags` - Number of lagged values to include

  ## Returns

    A tensor of shape `{n_predictions, lags}` with AR features.

  """
  @spec build_input(map(), list(), non_neg_integer()) :: Nx.Tensor.t()
  def build_input(training_data, prediction_dates, lags) do
    training_dates = training_data.dates
    training_y = training_data.y_normalized

    # Create a map from date to index for fast lookup
    date_to_idx =
      training_dates
      |> Enum.with_index()
      |> Map.new()

    # For each prediction date, get the lags previous y values
    ar_inputs =
      Enum.map(prediction_dates, fn date ->
        idx = Map.get(date_to_idx, date)

        if idx && idx >= lags do
          # Get y values from idx-lags to idx-1
          Enum.slice(training_y, (idx - lags)..(idx - 1))
        else
          # For dates at the beginning or not in training, use zeros
          List.duplicate(0.0, lags)
        end
      end)

    ar_inputs
    |> Nx.tensor()
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Extracts raw AR layer weights from a fitted model.

  For linear AR models, returns the output layer weights.
  For deep AR-Net models, returns all layer weights including hidden layers.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct with AR enabled.

  ## Returns

    A map of layer names to weight structs containing `:kernel` and `:bias` tensors.

  ## Examples

      iex> weights = Soothsayer.AR.get_weights(fitted_model)
      %{"ar_dense_out" => %{kernel: #Nx.Tensor<...>, bias: #Nx.Tensor<...>}}

  """
  @spec get_weights(Soothsayer.Model.t()) :: %{
          String.t() => %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}
        }
  def get_weights(%Soothsayer.Model{} = model) do
    unless model.config.ar.enabled do
      raise ArgumentError, "AR is not enabled on this model"
    end

    unless model.params do
      raise ArgumentError, "Model has not been fitted yet"
    end

    model.params.data
    |> Enum.filter(fn {name, _} -> String.starts_with?(name, "ar_dense") end)
    |> Enum.map(fn {name, layer} ->
      {name, %{kernel: layer["kernel"], bias: layer["bias"]}}
    end)
    |> Enum.into(%{})
  end
end
