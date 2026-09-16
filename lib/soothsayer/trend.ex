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

  alias Soothsayer.AR
  alias Soothsayer.Layers
  alias Soothsayer.Timestamp

  @seconds_per_day 86_400

  # Network Building

  @doc """
  Creates the Axon input node for the trend component.

  ## Parameters

    * `config` - Model configuration map with `:trend` key.

  ## Returns

    An Axon input node with shape `{nil, positions, 1 + changepoints}`,
    where `positions` is the number of timestamps in a sample, see
    `Soothsayer.AR.positions/1`.

  """
  @spec build_input(map()) :: Axon.t()
  def build_input(config) do
    changepoints = get_in(config, [:trend, :changepoints]) || 0
    Axon.input("trend", shape: {nil, AR.positions(config), 1 + changepoints})
  end

  @doc """
  Builds the trend component layer.

  ## Parameters

    * `input` - Axon input node from `build_input/1`.
    * `config` - Model configuration map.

  ## Returns

    A linear layer over every position, `{batch, positions}`, when enabled,
    or `Axon.constant(0)` when disabled.

  """
  @spec build_component(Axon.t(), map()) :: Axon.t()
  def build_component(input, %{trend: %{enabled: true}}) do
    Layers.position_dense(input, "trend_dense", use_bias: true)
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
    * `changepoints` - Number of changepoints to create.
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

  def compute_changepoint_indices(n_samples, changepoints, changepoints_range) do
    max_index = trunc(n_samples * changepoints_range)
    step = max_index / changepoints

    1..changepoints
    |> Enum.map(fn i -> trunc(i * step) end)
  end

  @doc """
  Computes changepoint positions as timestamps from the data.

  ## Parameters

    * `dates` - List of timestamps in the dataset.
    * `changepoints` - Number of changepoints to create.
    * `changepoints_range` - Fraction of data to place changepoints in (0-1).

  ## Returns

    A list of dates where changepoints are positioned.

  ## Examples

      iex> dates = Enum.map(0..99, fn i -> Date.add(~D[2023-01-01], i) end)
      iex> Soothsayer.Trend.compute_changepoint_positions(dates, 5, 0.8)
      [~D[2023-01-17], ~D[2023-02-02], ~D[2023-02-18], ~D[2023-03-06], ~D[2023-03-22]]

  """
  @spec compute_changepoint_positions(list(Timestamp.input()), non_neg_integer(), float()) ::
          list(Timestamp.input())
  def compute_changepoint_positions(_dates, 0, _changepoints_range), do: []

  def compute_changepoint_positions(dates, changepoints, changepoints_range) do
    n_samples = length(dates)
    indices = compute_changepoint_indices(n_samples, changepoints, changepoints_range)
    Enum.map(indices, fn idx -> Enum.at(dates, idx) end)
  end

  @doc """
  Builds changepoint feature tensor computing max(0, t - s_j) for each changepoint.

  ## Parameters

    * `t` - Tensor of time values with shape `{n_samples, 1}`.
    * `changepoint_positions` - List of numeric changepoint positions.

  ## Returns

    A tensor of shape `{n_samples, changepoints}` with changepoint features.

  ## Examples

      iex> t = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> Soothsayer.Trend.build_changepoint_features(t, [1.5])
      #Nx.Tensor<f32[3][1] [[0.0], [0.5], [1.5]]>

  """
  @spec build_changepoint_features(Nx.Tensor.t(), list(number())) :: Nx.Tensor.t() | nil
  def build_changepoint_features(_t, []), do: nil

  def build_changepoint_features(t, changepoint_positions) do
    t
    |> Nx.reshape({:auto, 1})
    |> Nx.subtract(Nx.tensor([changepoint_positions]))
    |> Nx.max(0)
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Builds the complete trend input by concatenating t with changepoint features.

  ## Parameters

    * `t` - Tensor of time values with shape `{n_samples, 1}`.
    * `changepoint_features` - Tensor of changepoint features with shape `{n_samples, changepoints}`.

  ## Returns

    A tensor of shape `{n_samples, 1 + changepoints}`.

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
  Converts timestamps to numeric values (days since the first timestamp,
  fractional for sub-daily data).

  ## Parameters

    * `timestamps` - List of dates or naive datetimes.
    * `first_timestamp` - Reference point (typically the first timestamp in the dataset).

  ## Returns

    A tensor of numeric values representing days since first_timestamp.

  ## Examples

      iex> Soothsayer.Trend.date_to_numeric([~D[2023-01-01], ~D[2023-01-02]], ~D[2023-01-01])
      #Nx.Tensor<f32[2] [0.0, 1.0]>

      iex> Soothsayer.Trend.date_to_numeric([~N[2023-01-01 06:00:00]], ~D[2023-01-01])
      #Nx.Tensor<f32[1] [0.25]>

  """
  @spec date_to_numeric(list(Timestamp.input()), Timestamp.input()) :: Nx.Tensor.t()
  def date_to_numeric(timestamps, first_timestamp) do
    timestamps
    |> Enum.map(&Timestamp.days_since(&1, first_timestamp))
    |> Nx.tensor()
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Builds trend features tensor and metadata from timestamps.

  ## Parameters

    * `timestamps` - List of dates or naive datetimes.
    * `config` - Model configuration map with `:trend` key.

  ## Returns

    A tuple `{tensor, metadata}` where:
    - `tensor` has shape `{n_timestamps, 1 + changepoints}`
    - `metadata` contains `:first_timestamp` and `:changepoint_positions`
      (in days since the first timestamp)

  ## Examples

      iex> dates = [~D[2023-01-01], ~D[2023-01-02], ~D[2023-01-03]]
      iex> config = %{trend: %{changepoints: 0, changepoints_range: 0.8}}
      iex> {tensor, metadata} = Soothsayer.Trend.build_features(dates, config)
      iex> Nx.shape(tensor)
      {3, 1}
      iex> metadata.first_timestamp
      ~D[2023-01-01]

  """
  @spec build_features(list(Timestamp.input()), map()) :: {Nx.Tensor.t(), map()}
  def build_features(timestamps, config) do
    first_timestamp = List.first(timestamps)
    changepoints = get_in(config, [:trend, :changepoints]) || 0
    changepoints_range = get_in(config, [:trend, :changepoints_range]) || 0.8

    changepoint_positions =
      compute_numeric_changepoint_positions(
        timestamps,
        first_timestamp,
        changepoints,
        changepoints_range
      )

    t = date_to_numeric(timestamps, first_timestamp) |> Nx.new_axis(-1)
    changepoint_features = build_changepoint_features(t, changepoint_positions)
    tensor = build_trend_input(t, changepoint_features)

    metadata = %{
      first_timestamp: first_timestamp,
      changepoint_positions: changepoint_positions
    }

    {tensor, metadata}
  end

  defp compute_numeric_changepoint_positions(
         timestamps,
         first_timestamp,
         changepoints,
         changepoints_range
       ) do
    timestamps
    |> compute_changepoint_positions(changepoints, changepoints_range)
    |> Enum.map(&Timestamp.days_since(&1, first_timestamp))
  end

  @doc """
  Converts numeric values (days since `first_timestamp`) back to timestamps,
  dates when the reference is a date and naive datetimes otherwise.

  ## Examples

      iex> Soothsayer.Trend.numeric_to_date([0.0, 1.0], ~D[2023-01-01])
      [~D[2023-01-01], ~D[2023-01-02]]

      iex> Soothsayer.Trend.numeric_to_date([0.5], ~N[2023-01-01 00:00:00])
      [~N[2023-01-01 12:00:00]]

  """
  @spec numeric_to_date(list(number()), Timestamp.input()) :: list(Timestamp.input())
  def numeric_to_date(numeric, %Date{} = first_date) do
    Enum.map(numeric, fn days -> Date.add(first_date, trunc(days)) end)
  end

  def numeric_to_date(numeric, %NaiveDateTime{} = first_timestamp) do
    Enum.map(numeric, fn days ->
      NaiveDateTime.add(first_timestamp, round(days * @seconds_per_day), :second)
    end)
  end
end
