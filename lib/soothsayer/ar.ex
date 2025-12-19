defmodule Soothsayer.AR do
  @moduledoc """
  Auto-regression (AR) component functionality.

  Handles creation of lagged inputs and extraction of AR weights from fitted models.
  """

  @doc """
  Creates lagged input features and corresponding targets for AR training.

  Given a time series y and number of lags, creates sliding windows where each
  window contains n_lags consecutive values, and the target is the next value.

  ## Parameters

    * `y` - A 1D tensor of time series values.
    * `n_lags` - Number of lagged values to use as features.

  ## Returns

    A tuple `{lagged, targets}` where:
    * `lagged` - Tensor of shape `{n_samples, n_lags}` with lagged features
    * `targets` - Tensor of shape `{n_samples, 1}` with target values

  ## Examples

      iex> y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {lagged, targets} = Soothsayer.AR.create_lagged_inputs(y, 3)
      iex> Nx.shape(lagged)
      {2, 3}

  """
  @spec create_lagged_inputs(Nx.Tensor.t(), non_neg_integer()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def create_lagged_inputs(y, n_lags) do
    y_list = Nx.to_flat_list(y)
    n = length(y_list)

    {lagged_list, target_list} =
      Enum.reduce((n_lags)..(n - 1), {[], []}, fn i, {lagged_acc, target_acc} ->
        window = Enum.slice(y_list, (i - n_lags)..(i - 1))
        target = Enum.at(y_list, i)
        {[window | lagged_acc], [target | target_acc]}
      end)

    lagged = lagged_list |> Enum.reverse() |> Nx.tensor() |> Nx.as_type({:f, 32})
    targets = target_list |> Enum.reverse() |> Nx.tensor() |> Nx.reshape({:auto, 1}) |> Nx.as_type({:f, 32})

    {lagged, targets}
  end

  @doc """
  Builds AR input tensor for prediction given training data and prediction dates.

  For each prediction date, looks up the previous n_lags values from the training
  data to use as AR features. Returns zeros for dates that don't have enough history.

  ## Parameters

    * `training_data` - Map with `:dates` (list of dates) and `:y_normalized` (list of values)
    * `prediction_dates` - List of dates to build AR inputs for
    * `n_lags` - Number of lagged values to include

  ## Returns

    A tensor of shape `{n_predictions, n_lags}` with AR features.

  """
  @spec build_input(map(), list(), non_neg_integer()) :: Nx.Tensor.t()
  def build_input(training_data, prediction_dates, n_lags) do
    training_dates = training_data.dates
    training_y = training_data.y_normalized

    # Create a map from date to index for fast lookup
    date_to_idx =
      training_dates
      |> Enum.with_index()
      |> Map.new()

    # For each prediction date, get the n_lags previous y values
    ar_inputs =
      Enum.map(prediction_dates, fn date ->
        idx = Map.get(date_to_idx, date)

        if idx && idx >= n_lags do
          # Get y values from idx-n_lags to idx-1
          Enum.slice(training_y, (idx - n_lags)..(idx - 1))
        else
          # For dates at the beginning or not in training, use zeros
          List.duplicate(0.0, n_lags)
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
  @spec get_weights(Soothsayer.Model.t()) :: %{String.t() => %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}}
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
