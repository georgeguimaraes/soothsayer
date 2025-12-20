defmodule Soothsayer.Trend do
  @moduledoc """
  Trend component with optional piecewise linear changepoints.

  Handles network building, feature engineering, and weight extraction for the trend.

  The trend function with changepoints is:
  ```
  trend(t) = k * t + m + sum(delta_j * max(0, t - s_j))
  ```

  Where:
  - `k` = base growth rate (learned)
  - `m` = offset (learned)
  - `s_j` = changepoint positions (computed from data, fixed)
  - `delta_j` = rate adjustments at each changepoint (learned)
  """

  # Network Building

  @doc """
  Creates the Axon input node for the trend component.

  ## Parameters

    * `config` - Model configuration map with `:trend` key.

  ## Returns

    An Axon input node with shape `{nil, 1 + n_changepoints}`.

  """
  @spec build_input(map()) :: Axon.t()
  def build_input(config) do
    n_changepoints = get_in(config, [:trend, :n_changepoints]) || 0
    Axon.input("trend", shape: {nil, 1 + n_changepoints})
  end

  @doc """
  Builds the trend component layer.

  ## Parameters

    * `input` - Axon input node from `build_input/1`.
    * `config` - Model configuration map.

  ## Returns

    An Axon dense layer when enabled, or `Axon.constant(0)` when disabled.

  """
  @spec build_component(Axon.t(), map()) :: Axon.t()
  def build_component(input, %{trend: %{enabled: true}}) do
    Axon.dense(input, 1, activation: :linear, name: "trend_dense")
  end

  def build_component(_input, _config), do: Axon.constant(0)

  @doc """
  Extracts learned trend weights from a fitted model.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct.

  ## Returns

    A map with `:kernel` and `:bias` tensors.

  """
  @spec get_weights(Soothsayer.Model.t()) :: %{kernel: Nx.Tensor.t(), bias: Nx.Tensor.t()}
  def get_weights(%Soothsayer.Model{} = model) do
    unless model.config.trend.enabled do
      raise ArgumentError, "Trend is not enabled on this model"
    end

    unless model.params do
      raise ArgumentError, "Model has not been fitted yet"
    end

    trend_layer = model.params.data["trend_dense"]

    unless trend_layer do
      raise ArgumentError, "Trend layer not found in model params"
    end

    %{kernel: trend_layer["kernel"], bias: trend_layer["bias"]}
  end

  # Feature Engineering

  @doc """
  Computes evenly spaced changepoint indices within the first portion of data.

  ## Parameters

    * `n_samples` - Total number of samples in the dataset.
    * `n_changepoints` - Number of changepoints to create.
    * `changepoints_range` - Fraction of data to place changepoints in (0-1).

  ## Returns

    A list of indices where changepoints will be placed.

  ## Examples

      iex> Soothsayer.Trend.compute_changepoint_indices(100, 5, 0.8)
      [16, 32, 48, 64, 80]

  """
  @spec compute_changepoint_indices(non_neg_integer(), non_neg_integer(), float()) ::
          list(non_neg_integer())
  def compute_changepoint_indices(_n_samples, 0, _changepoints_range), do: []

  def compute_changepoint_indices(n_samples, n_changepoints, changepoints_range) do
    max_index = trunc(n_samples * changepoints_range)
    step = max_index / n_changepoints

    1..n_changepoints
    |> Enum.map(fn i -> trunc(i * step) end)
  end

  @doc """
  Computes changepoint positions as dates from the data.

  ## Parameters

    * `dates` - List of dates in the dataset.
    * `n_changepoints` - Number of changepoints to create.
    * `changepoints_range` - Fraction of data to place changepoints in (0-1).

  ## Returns

    A list of dates where changepoints are positioned.

  ## Examples

      iex> dates = Enum.map(0..99, fn i -> Date.add(~D[2023-01-01], i) end)
      iex> Soothsayer.Trend.compute_changepoint_positions(dates, 5, 0.8)
      [~D[2023-01-17], ~D[2023-02-02], ~D[2023-02-18], ~D[2023-03-06], ~D[2023-03-22]]

  """
  @spec compute_changepoint_positions(list(Date.t()), non_neg_integer(), float()) ::
          list(Date.t())
  def compute_changepoint_positions(_dates, 0, _changepoints_range), do: []

  def compute_changepoint_positions(dates, n_changepoints, changepoints_range) do
    n_samples = length(dates)
    indices = compute_changepoint_indices(n_samples, n_changepoints, changepoints_range)
    Enum.map(indices, fn idx -> Enum.at(dates, idx) end)
  end

  @doc """
  Builds changepoint feature tensor computing max(0, t - s_j) for each changepoint.

  ## Parameters

    * `t` - Tensor of time values with shape `{n_samples, 1}`.
    * `changepoint_positions` - List of numeric changepoint positions.

  ## Returns

    A tensor of shape `{n_samples, n_changepoints}` with changepoint features.

  ## Examples

      iex> t = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> Soothsayer.Trend.build_changepoint_features(t, [1.5])
      #Nx.Tensor<f32[3][1] [[0.0], [0.5], [1.5]]>

  """
  @spec build_changepoint_features(Nx.Tensor.t(), list(number())) :: Nx.Tensor.t() | nil
  def build_changepoint_features(_t, []), do: nil

  def build_changepoint_features(t, changepoint_positions) do
    t_flat = Nx.flatten(t)

    changepoint_positions
    |> Enum.map(fn s_j ->
      t_flat
      |> Nx.subtract(s_j)
      |> Nx.max(0)
    end)
    |> Nx.stack(axis: 1)
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Builds the complete trend input by concatenating t with changepoint features.

  ## Parameters

    * `t` - Tensor of time values with shape `{n_samples, 1}`.
    * `changepoint_features` - Tensor of changepoint features with shape `{n_samples, n_changepoints}`.

  ## Returns

    A tensor of shape `{n_samples, 1 + n_changepoints}`.

  ## Examples

      iex> t = Nx.tensor([[1.0], [2.0]])
      iex> cp_features = Nx.tensor([[0.0, 0.0], [0.5, 0.0]])
      iex> Soothsayer.Trend.build_trend_input(t, cp_features)
      #Nx.Tensor<f32[2][3] [[1.0, 0.0, 0.0], [2.0, 0.5, 0.0]]>

  """
  @spec build_trend_input(Nx.Tensor.t(), Nx.Tensor.t() | nil) :: Nx.Tensor.t()
  def build_trend_input(t, nil), do: t

  def build_trend_input(t, changepoint_features) do
    Nx.concatenate([t, changepoint_features], axis: 1) |> Nx.as_type({:f, 32})
  end

  @doc """
  Converts dates to numeric values (days since first date).

  ## Parameters

    * `dates` - List of dates.
    * `first_date` - Reference date (typically first date in dataset).

  ## Returns

    A tensor of numeric values representing days since first_date.

  ## Examples

      iex> Soothsayer.Trend.date_to_numeric([~D[2023-01-01], ~D[2023-01-02]], ~D[2023-01-01])
      #Nx.Tensor<f32[2] [0.0, 1.0]>

  """
  @spec date_to_numeric(list(Date.t()), Date.t()) :: Nx.Tensor.t()
  def date_to_numeric(dates, first_date) do
    dates
    |> Enum.map(fn date -> Date.diff(date, first_date) * 1.0 end)
    |> Nx.tensor()
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Converts numeric values back to dates.

  ## Parameters

    * `numeric` - List of numeric values (days since first_date).
    * `first_date` - Reference date.

  ## Returns

    A list of dates.

  ## Examples

      iex> Soothsayer.Trend.numeric_to_date([0.0, 1.0], ~D[2023-01-01])
      [~D[2023-01-01], ~D[2023-01-02]]

  """
  @spec numeric_to_date(list(number()), Date.t()) :: list(Date.t())
  def numeric_to_date(numeric, first_date) do
    Enum.map(numeric, fn days ->
      Date.add(first_date, trunc(days))
    end)
  end
end
