defmodule Soothsayer.AR do
  @moduledoc """
  Auto-regression (AR) component functionality.

  Handles network building, feature engineering, and weight extraction for AR models.
  Supports both linear AR and deep AR-Net architectures with configurable hidden layers.

  ## Stationarized lags

  The AR network never sees the raw lags. As in NeuralProphet, it sees the
  lags minus the trend, seasonality, events and regressors evaluated at each
  lag's own timestamp, and models what those components leave over. That
  keeps the trend in charge of the level and the seasonalities of their
  cycles instead of letting the lags absorb everything, which is what makes
  multi-step forecasts follow the level rather than drift back to the
  training mean.

  To have those components available at the lag timestamps, every training
  sample is one forecast origin: its `lags` lag timestamps followed by its
  `forecast_steps` target timestamps, see `training_samples/4`. The
  time-based inputs are `{samples, positions, features}` tensors over those
  positions and the network's outputs are `{samples, forecast_steps}`.
  """

  alias Soothsayer.Frequency
  alias Soothsayer.Timestamp

  # Network Building

  @doc """
  Creates the Axon input node for the AR component, `{nil, lags}`.

  Returns `nil` when AR is disabled or has no lags.
  """
  @spec build_network_input(map()) :: Axon.t() | nil
  def build_network_input(config) do
    case lags(config) do
      0 -> nil
      lags -> Axon.input("ar", shape: {nil, lags})
    end
  end

  @doc """
  Builds the AR component layer(s).

  The lags are stationarized first: `nonstationary_at_lags`, the other
  components summed at the lag positions (`{batch, lags}`), is subtracted
  from the input. Then come the hidden layers of a deep AR-Net (ReLU), if
  any, and a linear output layer with one unit per forecast step and no
  bias, as in NeuralProphet: a bias there would let the AR carry a level
  that belongs to the trend.

  ## Parameters

    * `input` - Axon input node from `build_network_input/1`.
    * `nonstationary_at_lags` - Axon node with the other components summed
      at the lag positions, `Axon.constant(0)` when there are none.
    * `config` - Model configuration map.

  ## Returns

    An Axon layer with `forecast_steps` outputs when AR is enabled,
    `Axon.constant(0)` otherwise.

  """
  @spec build_component(Axon.t() | nil, Axon.t(), map()) :: Axon.t()
  def build_component(nil, _nonstationary_at_lags, _config), do: Axon.constant(0)

  def build_component(input, nonstationary_at_lags, %{ar: %{enabled: true} = ar_config} = config) do
    layers = Map.get(ar_config, :layers, [])

    input
    |> Axon.subtract(nonstationary_at_lags, name: "ar_stationarized")
    |> build_hidden_layers(layers)
    |> Axon.dense(forecast_steps(config),
      activation: :linear,
      name: "ar_dense_out",
      use_bias: false
    )
  end

  def build_component(_input, _nonstationary_at_lags, _config), do: Axon.constant(0)

  @doc """
  The number of lags the AR component uses, 0 when it is disabled.
  """
  @spec lags(map()) :: non_neg_integer()
  def lags(%{ar: %{enabled: true, lags: lags}}) when is_integer(lags) and lags > 0, do: lags
  def lags(_config), do: 0

  @doc """
  Returns the configured number of direct forecast steps, defaulting to 1.

  With `forecast_steps: k` the AR output layer has `k` units, one per step
  ahead, and every training sample holds the `k` targets after its origin.
  This is NeuralProphet's `n_forecasts`.
  """
  @spec forecast_steps(map()) :: pos_integer()
  def forecast_steps(%{ar: %{forecast_steps: steps}}) when is_integer(steps) and steps > 0 do
    steps
  end

  def forecast_steps(_config), do: 1

  @doc """
  The number of timestamps in one sample: the `lags/1` lag positions
  followed by the `forecast_steps/1` target positions. One without
  auto-regression.
  """
  @spec positions(map()) :: pos_integer()
  def positions(config), do: lags(config) + forecast_steps(config)

  defp build_hidden_layers(input, layers) do
    {hidden, _idx} =
      Enum.reduce(layers, {input, 0}, fn units, {acc, idx} ->
        {Axon.dense(acc, units, activation: :relu, name: "ar_dense_#{idx}"), idx + 1}
      end)

    hidden
  end

  # Feature Engineering

  @doc """
  Creates lagged input features and corresponding targets for one-step AR training.

  Equivalent to `training_samples(y, lags, 1)`, kept for its simpler shape.

  ## Examples

      iex> y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {lagged, targets} = Soothsayer.AR.create_lagged_inputs(y, 3)
      iex> Nx.shape(lagged)
      {2, 3}

  """
  @spec create_lagged_inputs(Nx.Tensor.t(), non_neg_integer()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def create_lagged_inputs(y, lags) do
    %{lagged: lagged, targets: targets} = training_samples(y, lags, 1)
    {lagged, targets}
  end

  @doc """
  Builds the AR training samples for direct multi-step forecasting.

  Every origin (a position with `max_lags` values up to and including it and
  `forecast_steps` values after it) is one sample: the `lags` values ending
  at the origin, oldest first, and the `forecast_steps` values after it as
  its targets. A sample's positions are its lag timestamps followed by its
  target timestamps, and `position_indices` says where each of them sits in
  `y`, so the time-based features can be gathered per sample.

  ## Options

    * `:max_lags` - the longest lag window any input needs, which decides
      the first usable origin. Defaults to `lags`. Lagged regressors with
      more lags than the AR component raise it.
    * `:skip_positions` - a `MapSet` of positions in `y` that are missing.
      An origin is left out when its widest window, `max_lags` values up
      to and including it and `forecast_steps` after it, touches one of
      them. Raises when no origin is left.

  ## Returns

    A map with:
    * `:lagged` - `{samples, lags}` lag values
    * `:targets` - `{samples, forecast_steps}` target values
    * `:position_indices` - `{samples, lags + forecast_steps}` positions in `y`
    * `:origin_indices` - list of the origin positions
    * `:skipped_origins` - how many origins `:skip_positions` left out

  ## Examples

      iex> y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      iex> samples = Soothsayer.AR.training_samples(y, 2, 2)
      iex> Nx.to_list(samples.lagged)
      [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
      iex> Nx.to_list(samples.targets)
      [[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
      iex> Nx.to_list(samples.position_indices)
      [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
      iex> samples.origin_indices
      [1, 2, 3]

      iex> y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      iex> samples = Soothsayer.AR.training_samples(y, 2, 2, skip_positions: MapSet.new([4]))
      iex> {samples.origin_indices, samples.skipped_origins}
      {[1], 2}

  """
  @spec training_samples(Nx.Tensor.t(), pos_integer(), pos_integer(), keyword()) :: %{
          lagged: Nx.Tensor.t(),
          targets: Nx.Tensor.t(),
          position_indices: Nx.Tensor.t(),
          origin_indices: list(non_neg_integer()),
          skipped_origins: non_neg_integer()
        }
  def training_samples(y, lags, forecast_steps, opts \\ []) do
    max_lags = Keyword.get(opts, :max_lags, lags)
    skip_positions = Keyword.get(opts, :skip_positions, MapSet.new())
    all_origins = origin_indices(Nx.size(y), max_lags, forecast_steps)

    origin_indices =
      Enum.reject(all_origins, &touches?(&1, skip_positions, max_lags, forecast_steps))

    if origin_indices == [] do
      raise ArgumentError,
            "Every training sample touches a missing value. Fill the gaps before fitting."
    end

    lag_indices = Enum.map(origin_indices, &window_indices(&1, lags))

    target_indices =
      for origin <- origin_indices, do: Enum.map(1..forecast_steps, &(origin + &1))

    %{
      lagged: y |> Nx.take(Nx.tensor(lag_indices)) |> Nx.as_type({:f, 32}),
      targets: y |> Nx.take(Nx.tensor(target_indices)) |> Nx.as_type({:f, 32}),
      position_indices: Enum.zip_with(lag_indices, target_indices, &Kernel.++/2) |> Nx.tensor(),
      origin_indices: origin_indices,
      skipped_origins: length(all_origins) - length(origin_indices)
    }
  end

  defp touches?(origin, skip_positions, max_lags, forecast_steps) do
    MapSet.size(skip_positions) > 0 and
      Enum.any?(
        (origin - max_lags + 1)..(origin + forecast_steps),
        &MapSet.member?(skip_positions, &1)
      )
  end

  @doc """
  The usable origin positions in a series of `n` values: every position with
  `max_lags` values up to and including it and `forecast_steps` after it.

  ## Examples

      iex> Soothsayer.AR.origin_indices(10, 4, 2)
      [3, 4, 5, 6, 7]

  """
  @spec origin_indices(pos_integer(), pos_integer(), pos_integer()) :: list(non_neg_integer())
  def origin_indices(n, max_lags, forecast_steps) do
    if n < max_lags + forecast_steps do
      raise ArgumentError,
            "Not enough data for #{max_lags} lags and #{forecast_steps} forecast steps: " <>
              "need at least #{max_lags + forecast_steps} rows, got #{n}"
    end

    Enum.to_list((max_lags - 1)..(n - forecast_steps - 1))
  end

  @doc """
  Lag windows of `values` ending at each origin, oldest first, `{origins, lags}`.

  ## Examples

      iex> Soothsayer.AR.lagged_rows(Nx.tensor([10.0, 20.0, 30.0, 40.0]), [2, 3], 2) |> Nx.to_list()
      [[20.0, 30.0], [30.0, 40.0]]

  """
  @spec lagged_rows(Nx.Tensor.t(), list(non_neg_integer()), pos_integer()) :: Nx.Tensor.t()
  def lagged_rows(values, origin_indices, lags) do
    indices = Enum.map(origin_indices, &window_indices(&1, lags))
    values |> Nx.take(Nx.tensor(indices)) |> Nx.as_type({:f, 32})
  end

  defp window_indices(origin, lags), do: Enum.map((lags - 1)..0//-1, &(origin - &1))

  @doc """
  The timestamps of one prediction sample from `origin`: the `lags`
  timestamps ending at the origin, oldest first, then the `forecast_steps`
  timestamps after it, all one step of `frequency` apart.

  ## Examples

      iex> Soothsayer.AR.sample_timestamps(~D[2023-01-10], 2, 2, {1, :day})
      [~D[2023-01-09], ~D[2023-01-10], ~D[2023-01-11], ~D[2023-01-12]]

  """
  @spec sample_timestamps(Timestamp.input(), non_neg_integer(), pos_integer(), Frequency.t()) ::
          list(Timestamp.input())
  def sample_timestamps(origin, lags, forecast_steps, frequency) do
    Enum.map((1 - lags)..forecast_steps, &Frequency.shift(origin, &1, frequency))
  end

  @doc """
  Decides which origin and step a prediction timestamp is forecast from.

  Timestamps up to the last observed one are one step ahead of the step
  before them. Later timestamps are forecast in blocks of `forecast_steps`:
  the first block from the last observation, the next block from the end of
  the first block, and so on. That is how NeuralProphet's maintainers
  recommend going past `n_forecasts`. Steps are steps of `frequency`, and a
  timestamp off that grid raises `ArgumentError`.

  ## Examples

      iex> Soothsayer.AR.origin_and_step(~D[2023-01-10], ~D[2023-01-31], 3, {1, :day})
      {~D[2023-01-09], 1}
      iex> Soothsayer.AR.origin_and_step(~D[2023-02-03], ~D[2023-01-31], 3, {1, :day})
      {~D[2023-01-31], 3}
      iex> Soothsayer.AR.origin_and_step(~D[2023-02-04], ~D[2023-01-31], 3, {1, :day})
      {~D[2023-02-03], 1}
      iex> Soothsayer.AR.origin_and_step(~N[2023-01-01 02:00:00], ~N[2023-01-01 00:00:00], 3, {1, :hour})
      {~N[2023-01-01 00:00:00], 2}

  """
  @spec origin_and_step(Timestamp.input(), Timestamp.input(), pos_integer(), Frequency.t()) ::
          {Timestamp.input(), pos_integer()}
  def origin_and_step(timestamp, last_observed_timestamp, forecast_steps, frequency) do
    distance = Frequency.steps_between(last_observed_timestamp, timestamp, frequency)

    if distance <= 0 do
      {Frequency.shift(timestamp, -1, frequency), 1}
    else
      block = div(distance - 1, forecast_steps)
      origin = Frequency.shift(last_observed_timestamp, block * forecast_steps, frequency)
      {origin, distance - block * forecast_steps}
    end
  end

  @doc """
  Builds a map of known values keyed by timestamp from the data stored at fit time.

  Values are in normalized y space, the same space the network predicts in.

  ## Parameters

    * `training_data` - Map with `:timestamps` and `:y_normalized` (list of
      values). A fitted model also carries the zipped map as `:known_values`,
      which is returned as is.

  ## Returns

    A map from timestamp to the normalized value observed then.

  """
  @spec known_values(%{
          optional(:known_values) => %{Timestamp.input() => float()},
          timestamps: list(Timestamp.input()),
          y_normalized: list(float())
        }) ::
          %{Timestamp.input() => float()}
  def known_values(%{known_values: %{} = known_values}), do: known_values

  def known_values(%{timestamps: timestamps, y_normalized: y_normalized}) do
    Enum.zip(timestamps, y_normalized) |> Map.new()
  end

  @doc """
  Builds the AR input tensor for a list of origin timestamps.

  For each origin, looks up the `lags` values ending at it (the origin
  itself and the steps before it), oldest first, matching the column order
  of `training_samples/4`. Origins where any of those steps is unknown get a
  row of zeros.

  ## Parameters

    * `known_values` - Map from timestamp to normalized value, see `known_values/1`
    * `origins` - List of origin timestamps, one per row
    * `lags` - Number of lagged values to include
    * `frequency` - The step between lags, see `Soothsayer.Frequency`

  ## Examples

      iex> known_values = %{~D[2023-01-01] => 1.0, ~D[2023-01-02] => 2.0, ~D[2023-01-03] => 3.0}
      iex> Soothsayer.AR.build_input(known_values, [~D[2023-01-02], ~D[2023-01-03]], 2, {1, :day})
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 2.0],
          [2.0, 3.0]
        ]
      >

  """
  @spec build_input(
          %{Timestamp.input() => float()},
          list(Timestamp.input()),
          non_neg_integer(),
          Frequency.t()
        ) :: Nx.Tensor.t()
  def build_input(known_values, origins, lags, frequency) do
    rows =
      Enum.map(origins, fn origin ->
        lagged_values =
          Enum.map((lags - 1)..0//-1, fn offset ->
            Map.get(known_values, Frequency.shift(origin, -offset, frequency))
          end)

        if Enum.any?(lagged_values, &is_nil/1) do
          List.duplicate(0.0, lags)
        else
          lagged_values
        end
      end)

    rows
    |> Nx.tensor()
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Extracts raw AR layer weights from a fitted model.

  For linear AR models, returns the output layer weights.
  For deep AR-Net models, returns all layer weights including hidden layers.

  The output kernel has shape `{inputs, forecast_steps}`: row `i` is the
  i-th oldest lag and column `s` holds the weights for step `s + 1` ahead.
  Hidden layers have a `:bias` as well; the output layer has none.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct with AR enabled.

  ## Returns

    A map of layer names to maps with a `:kernel` tensor and, for hidden
    layers, a `:bias` tensor.

  ## Examples

      iex> weights = Soothsayer.AR.get_weights(fitted_model)
      %{"ar_dense_out" => %{kernel: #Nx.Tensor<...>}}

  """
  @spec get_weights(Soothsayer.Model.t()) :: %{
          String.t() => %{optional(:bias) => Nx.Tensor.t(), kernel: Nx.Tensor.t()}
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
    |> Map.new(fn {name, layer} ->
      {name, Map.reject(%{kernel: layer["kernel"], bias: layer["bias"]}, &is_nil(elem(&1, 1)))}
    end)
  end
end
