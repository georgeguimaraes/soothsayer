defmodule Soothsayer.Trend do
  @moduledoc """
  Trend component with optional piecewise linear changepoints.

  Handles network building, feature engineering, and weight extraction for the trend.

  The trend function with changepoints is:
  ```
  trend(t) = k * t + m + sum(delta_j * f_j(t))
  ```

  Where:
  - `k` = base growth rate (learned)
  - `m` = offset (learned)
  - `s_j` = changepoint positions (computed from data, fixed)
  - `delta_j` = slope adjustments (learned)
  - `f_j` = the changepoint basis, see `basis/1`

  Without trend regularization the basis is segmentwise, like
  NeuralProphet's: `f_j(t) = min(max(0, t - s_j), s_{j+1} - s_j)`, a hinge
  that stops growing at the next changepoint, so `delta_j` is the slope of
  segment `j` relative to `k` and each segment fits its own data. With
  regularization the basis is the cumulative Prophet hinge
  `f_j(t) = max(0, t - s_j)`, where `delta_j` is the change of slope at
  `s_j`, which is what an L1 penalty on the deltas should shrink.

  With `growth: :discontinuous` the trend may also jump at each changepoint,
  NeuralProphet's discontinuous growth: one more learned intercept per
  segment after the first, as extra input columns after the slope columns.
  Under the segmentwise basis those are one-hot segment indicators and the
  slope columns become ramps that live only inside their segment, so each
  segment's slope and level are trained by its own data. Under the
  cumulative basis they are steps `1[t >= s_j]` next to the cumulative
  hinges, so the L1 penalty means few jumps. `growth: :linear` (the default)
  keeps the trend continuous. NeuralProphet's `growth: "off"` is
  `trend: %{enabled: false}` here.
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

    An Axon input node with shape `{nil, positions, features}`, where
    `positions` is the number of timestamps in a sample, see
    `Soothsayer.AR.positions/1`, and `features` is `feature_count/1`.

  """
  @spec build_input(map()) :: Axon.t()
  def build_input(config) do
    Axon.input("trend", shape: {nil, AR.positions(config), feature_count(config)})
  end

  @doc """
  The growth of a config's trend, `:linear` unless it says `:discontinuous`.
  """
  @spec growth(map()) :: :linear | :discontinuous
  def growth(config), do: get_in(config, [:trend, :growth]) || :linear

  @doc """
  How many leading trend input columns are measured in time: `t` and one
  slope column per changepoint. Fit scales those by the training span. The
  intercept columns of discontinuous growth come after them and stay as
  they are.
  """
  @spec time_columns(map()) :: pos_integer()
  def time_columns(config), do: 1 + (get_in(config, [:trend, :changepoints]) || 0)

  @doc """
  The width of the trend input: `time_columns/1` plus one intercept column
  per changepoint with discontinuous growth.

  ## Examples

      iex> Soothsayer.Trend.feature_count(%{trend: %{changepoints: 10}})
      11

      iex> Soothsayer.Trend.feature_count(%{trend: %{changepoints: 10, growth: :discontinuous}})
      21

  """
  @spec feature_count(map()) :: pos_integer()
  def feature_count(config) do
    changepoints = get_in(config, [:trend, :changepoints]) || 0

    case growth(config) do
      :discontinuous -> time_columns(config) + changepoints
      :linear -> time_columns(config)
    end
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
  def build_component(input, config, series_input \\ nil)

  def build_component(input, %{trend: %{enabled: true}} = config, series_input) do
    case Soothsayer.Series.local_ids(config, :trend) do
      nil ->
        Layers.position_dense(input, "trend_dense", use_bias: true)

      ids ->
        Layers.series_dense(input, series_input, length(ids), "trend_dense", use_bias: true)
    end
  end

  def build_component(_input, _config, _series_input), do: Axon.constant(0)

  @doc """
  Extracts learned trend weights from a fitted model.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct.

  ## Returns

    A map with `:kernel` and `:bias` tensors. The kernel has one row per
    trend input column: `t`, then one slope adjustment per changepoint,
    then, with `growth: :discontinuous`, one intercept per changepoint. The
    bias is the offset `m`.

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

    weights = %{kernel: trend_layer["kernel"], bias: trend_layer["bias"]}

    # A local trend keeps one kernel and bias per series, leading axis
    case Soothsayer.Series.local_ids(model.config, :trend) do
      nil -> weights
      ids -> Soothsayer.Series.by_id(ids, weights)
    end
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
      [13, 26, 40, 53, 66]

  """
  @spec compute_changepoint_indices(non_neg_integer(), non_neg_integer(), float()) ::
          list(non_neg_integer())
  def compute_changepoint_indices(_n_samples, 0, _changepoints_range), do: []

  def compute_changepoint_indices(n_samples, changepoints, changepoints_range) do
    # NeuralProphet spreads n + 1 points evenly over the first
    # changepoints_range of the data, the first at zero, so the last
    # changepoint sits at range * n / (n + 1) and the final segment, the
    # one every forecast extrapolates, is a little longer than the others.
    max_index = n_samples * changepoints_range
    step = max_index / (changepoints + 1)

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
  The changepoint basis a config trains with: `:segmentwise` without trend
  regularization, `:cumulative` with it. See the module docs.

  ## Examples

      iex> Soothsayer.Trend.basis(%{trend: %{regularization: nil}})
      :segmentwise

      iex> Soothsayer.Trend.basis(%{trend: %{regularization: 0.1}})
      :cumulative

  """
  @spec basis(map()) :: :segmentwise | :cumulative
  def basis(%{trend: %{regularization: regularization}}) when not is_nil(regularization),
    do: :cumulative

  def basis(_config), do: :segmentwise

  @doc """
  Builds the changepoint feature tensor, one column per changepoint.

  ## Parameters

    * `t` - Tensor of time values with shape `{n_samples, 1}`.
    * `changepoint_positions` - List of numeric changepoint positions, ascending.
    * `basis` - `:cumulative` (default) for `max(0, t - s_j)`, or
      `:segmentwise` to clip each hinge at the next changepoint. See `basis/1`.
    * `growth` - `:linear` (default), or `:discontinuous` to append one
      intercept column per changepoint: one-hot segment indicators with
      ramps for slopes under `:segmentwise`, steps `1[t >= s_j]` under
      `:cumulative`. See the module docs.

  ## Returns

    A tensor of shape `{n_samples, changepoints}` with changepoint features,
    `{n_samples, 2 * changepoints}` with discontinuous growth.

  ## Examples

      iex> t = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> Soothsayer.Trend.build_changepoint_features(t, [1.5])
      #Nx.Tensor<f32[3][1] [[0.0], [0.5], [1.5]]>

      iex> t = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> Soothsayer.Trend.build_changepoint_features(t, [1.5, 2.5], :segmentwise)
      #Nx.Tensor<f32[3][2] [[0.0, 0.0], [0.5, 0.0], [1.0, 0.5]]>

      iex> t = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> Soothsayer.Trend.build_changepoint_features(t, [1.5, 2.5], :segmentwise, :discontinuous) |> Nx.to_list()
      [[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0]]

  """
  @spec build_changepoint_features(
          Nx.Tensor.t(),
          list(number()),
          :cumulative | :segmentwise,
          :linear | :discontinuous
        ) :: Nx.Tensor.t() | nil
  def build_changepoint_features(
        t,
        changepoint_positions,
        basis \\ :cumulative,
        growth \\ :linear
      )

  def build_changepoint_features(_t, [], _basis, _growth), do: nil

  def build_changepoint_features(t, changepoint_positions, basis, growth) do
    t = Nx.reshape(t, {:auto, 1})
    starts = Nx.tensor([changepoint_positions])
    widths = Nx.tensor([segment_widths(changepoint_positions)])
    hinges = t |> Nx.subtract(starts) |> Nx.max(0)

    in_segment =
      Nx.logical_and(Nx.greater_equal(t, starts), Nx.less(t, Nx.add(starts, widths)))
      |> Nx.as_type({:f, 32})

    slopes =
      case {basis, growth} do
        {:cumulative, _growth} -> hinges
        {:segmentwise, :linear} -> Nx.min(hinges, widths)
        {:segmentwise, :discontinuous} -> Nx.multiply(hinges, in_segment)
      end

    intercepts =
      case {basis, growth} do
        {_basis, :linear} -> nil
        {:cumulative, :discontinuous} -> Nx.greater_equal(t, starts)
        {:segmentwise, :discontinuous} -> in_segment
      end

    [slopes, intercepts]
    |> Enum.reject(&is_nil/1)
    |> Enum.map(&Nx.as_type(&1, {:f, 32}))
    |> Nx.concatenate(axis: 1)
  end

  # The last segment has no end, so its hinge keeps growing into the future.
  defp segment_widths(changepoint_positions) do
    changepoint_positions
    |> Enum.chunk_every(2, 1, [:infinity])
    |> Enum.map(fn
      [_start, :infinity] -> :infinity
      [start, next] -> next - start
    end)
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
    %{first_timestamp: first_timestamp, changepoint_positions: changepoint_positions} =
      metadata = changepoint_metadata(timestamps, config)

    t = date_to_numeric(timestamps, first_timestamp) |> Nx.new_axis(-1)

    changepoint_features =
      build_changepoint_features(t, changepoint_positions, basis(config), growth(config))

    {build_trend_input(t, changepoint_features), metadata}
  end

  @doc """
  The time axis of a model: `first_timestamp`, the origin of the numeric
  time, and `changepoint_positions`, in days from it, spread over the first
  `changepoints_range` of the sorted timestamps.
  """
  @spec changepoint_metadata(list(Timestamp.input()), map()) :: %{
          first_timestamp: Timestamp.input(),
          changepoint_positions: list(float()) | nil
        }
  def changepoint_metadata(timestamps, config) do
    first_timestamp = List.first(timestamps)
    changepoints = get_in(config, [:trend, :changepoints]) || 0
    changepoints_range = get_in(config, [:trend, :changepoints_range]) || 0.8

    %{
      first_timestamp: first_timestamp,
      changepoint_positions:
        compute_numeric_changepoint_positions(
          timestamps,
          first_timestamp,
          changepoints,
          changepoints_range
        )
    }
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
