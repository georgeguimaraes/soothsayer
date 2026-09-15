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
  def build_component(input, config) do
    build_component(input, build_step_mask_input(config), config)
  end

  @doc """
  Builds the AR component layer(s) with an explicit step mask input node.

  Use this when the step mask node is shared with other parts of the
  network, see `build_step_mask_input/1`.
  """
  @spec build_component(Axon.t() | nil, Axon.t() | nil, map()) :: Axon.t()
  def build_component(nil, _step_mask_input, _config), do: Axon.constant(0)

  def build_component(input, step_mask_input, %{ar: %{enabled: true} = ar_config} = config) do
    layers = Map.get(ar_config, :layers, [])
    steps = forecast_steps(config)

    input
    |> build_hidden_layers(layers)
    |> Axon.dense(steps, activation: :linear, name: "ar_dense_out")
    |> select_forecast_step(step_mask_input)
  end

  def build_component(_input, _step_mask_input, _config), do: Axon.constant(0)

  @doc """
  Creates the one-hot `"forecast_step"` input node, `{nil, forecast_steps}`.

  Returns `nil` when AR is disabled or `forecast_steps` is 1, since a single
  step needs no selection.
  """
  @spec build_step_mask_input(map()) :: Axon.t() | nil
  def build_step_mask_input(%{ar: %{enabled: true}} = config) do
    case forecast_steps(config) do
      1 -> nil
      steps -> Axon.input("forecast_step", shape: {nil, steps})
    end
  end

  def build_step_mask_input(_config), do: nil

  @doc """
  Returns the configured number of direct forecast steps, defaulting to 1.

  With `forecast_steps: k` the AR output layer has `k` units, one per step
  ahead, and each training row or prediction picks its step with a one-hot
  `"forecast_step"` input. This is NeuralProphet's `n_forecasts`.
  """
  @spec forecast_steps(map()) :: pos_integer()
  def forecast_steps(%{ar: %{forecast_steps: steps}}) when is_integer(steps) and steps > 0 do
    steps
  end

  def forecast_steps(_config), do: 1

  # With a single step the dense layer already outputs {batch, 1}. With more,
  # the {batch, steps} output is masked down to the row's own step.
  defp select_forecast_step(output, nil), do: output

  defp select_forecast_step(output, step_mask_input) do
    output
    |> Axon.multiply(step_mask_input)
    |> Axon.nx(&Nx.sum(&1, axes: [1], keep_axes: true), name: "ar_step_select")
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
  Creates lagged input features and corresponding targets for one-step AR training.

  Equivalent to `training_rows(y, lags, 1)`, kept for its simpler shape.

  ## Examples

      iex> y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> {lagged, targets} = Soothsayer.AR.create_lagged_inputs(y, 3)
      iex> Nx.shape(lagged)
      {2, 3}

  """
  @spec create_lagged_inputs(Nx.Tensor.t(), non_neg_integer()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def create_lagged_inputs(y, lags) do
    %{lagged: lagged, targets: targets} = training_rows(y, lags, 1)
    {lagged, targets}
  end

  @doc """
  Builds the AR training rows for direct multi-step forecasting.

  Every origin (a position with `max_lags` values up to and including it and
  `forecast_steps` values after it) produces one row per step ahead. A row
  holds the `lags` values ending at the origin, oldest first, the target
  `step` positions after the origin, and a one-hot mask of its step. Rows
  are ordered step by step, all origins for step 1 first, then all origins
  for step 2, and so on.

  ## Options

    * `:max_lags` - the longest lag window any input needs, which decides
      the first usable origin. Defaults to `lags`. Lagged regressors with
      more lags than the AR component raise it.

  ## Returns

    A map with:
    * `:lagged` - `{rows, lags}` lag values
    * `:targets` - `{rows, 1}` target values
    * `:step_mask` - `{rows, forecast_steps}` one-hot masks, or `nil` when `forecast_steps` is 1
    * `:target_indices` - list of the target position of each row in `y`, for
      lining up the date-based features
    * `:origin_indices` - list of the origin positions, one per origin (not per row)

  ## Examples

      iex> y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      iex> rows = Soothsayer.AR.training_rows(y, 2, 2)
      iex> Nx.to_list(rows.lagged)
      [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
      iex> Nx.to_flat_list(rows.targets)
      [3.0, 4.0, 5.0, 4.0, 5.0, 6.0]
      iex> rows.target_indices
      [2, 3, 4, 3, 4, 5]

  """
  @spec training_rows(Nx.Tensor.t(), pos_integer(), pos_integer(), keyword()) :: %{
          lagged: Nx.Tensor.t(),
          targets: Nx.Tensor.t(),
          step_mask: Nx.Tensor.t() | nil,
          target_indices: list(non_neg_integer()),
          origin_indices: list(non_neg_integer())
        }
  def training_rows(y, lags, forecast_steps, opts \\ []) do
    max_lags = Keyword.get(opts, :max_lags, lags)
    origin_indices = origin_indices(Nx.size(y), max_lags, forecast_steps)

    target_indices =
      for step <- 1..forecast_steps, origin <- origin_indices, do: origin + step

    targets =
      y
      |> Nx.take(Nx.tensor(target_indices))
      |> Nx.reshape({:auto, 1})
      |> Nx.as_type({:f, 32})

    step_numbers = for step <- 1..forecast_steps, _origin <- origin_indices, do: step

    %{
      lagged: lagged_rows(y, origin_indices, lags, forecast_steps),
      targets: targets,
      step_mask: step_mask(step_numbers, forecast_steps),
      target_indices: target_indices,
      origin_indices: origin_indices
    }
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
  Lag windows of `values` ending at each origin, oldest first, `{origins, lags}`,
  repeated `forecast_steps` times along the row axis to match `training_rows/4`.

  ## Examples

      iex> Soothsayer.AR.lagged_rows(Nx.tensor([10.0, 20.0, 30.0, 40.0]), [2, 3], 2, 1) |> Nx.to_list()
      [[20.0, 30.0], [30.0, 40.0]]

  """
  @spec lagged_rows(Nx.Tensor.t(), list(non_neg_integer()), pos_integer(), pos_integer()) ::
          Nx.Tensor.t()
  def lagged_rows(values, origin_indices, lags, forecast_steps) do
    window_indices =
      for origin <- origin_indices, do: for(offset <- (lags - 1)..0//-1, do: origin - offset)

    windows = values |> Nx.take(Nx.tensor(window_indices)) |> Nx.as_type({:f, 32})

    Nx.concatenate(List.duplicate(windows, forecast_steps), axis: 0)
  end

  @doc """
  One-hot encodes step numbers (1-based) into a `{rows, forecast_steps}` mask.

  Returns `nil` when `forecast_steps` is 1, since the network has no
  `"forecast_step"` input in that case.

  ## Examples

      iex> Soothsayer.AR.step_mask([1, 3], 3) |> Nx.to_list()
      [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

  """
  @spec step_mask(list(pos_integer()), pos_integer()) :: Nx.Tensor.t() | nil
  def step_mask(_step_numbers, 1), do: nil

  def step_mask(step_numbers, forecast_steps) do
    step_numbers
    |> Enum.map(&(&1 - 1))
    |> Nx.tensor()
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.iota({forecast_steps}))
    |> Nx.as_type({:f, 32})
  end

  @doc """
  Decides which origin and step a prediction date is forecast from.

  Dates up to the last observed date are one step ahead of the day before
  them. Later dates are forecast in blocks of `forecast_steps`: the first
  block from the last observed date, the next block from the last date of the
  first block, and so on. That is how NeuralProphet's maintainers recommend
  going past `n_forecasts`.

  ## Examples

      iex> Soothsayer.AR.origin_and_step(~D[2023-01-10], ~D[2023-01-31], 3)
      {~D[2023-01-09], 1}
      iex> Soothsayer.AR.origin_and_step(~D[2023-02-03], ~D[2023-01-31], 3)
      {~D[2023-01-31], 3}
      iex> Soothsayer.AR.origin_and_step(~D[2023-02-04], ~D[2023-01-31], 3)
      {~D[2023-02-03], 1}

  """
  @spec origin_and_step(Date.t(), Date.t(), pos_integer()) :: {Date.t(), pos_integer()}
  def origin_and_step(date, last_observed_date, forecast_steps) do
    distance = Date.diff(date, last_observed_date)

    if distance <= 0 do
      {Date.add(date, -1), 1}
    else
      block = div(distance - 1, forecast_steps)
      origin_date = Date.add(last_observed_date, block * forecast_steps)
      {origin_date, distance - block * forecast_steps}
    end
  end

  @doc """
  Builds a map of known values keyed by date from the data stored at fit time.

  Values are in normalized y space, the same space the network predicts in.

  ## Parameters

    * `training_data` - Map with `:dates` (list of dates) and `:y_normalized` (list of values)

  ## Returns

    A map from `Date.t()` to the normalized value observed on that date.

  """
  @spec known_values(%{dates: list(Date.t()), y_normalized: list(float())}) ::
          %{Date.t() => float()}
  def known_values(%{dates: dates, y_normalized: y_normalized}) do
    Enum.zip(dates, y_normalized) |> Map.new()
  end

  @doc """
  Builds the AR input tensor for a list of origin dates.

  For each origin, looks up the `lags` values ending on that day (the origin
  itself and the days before it), oldest first, matching the column order
  of `training_rows/3`. Origins where any of those days is unknown get a row
  of zeros.

  ## Parameters

    * `known_values` - Map from `Date.t()` to normalized value, see `known_values/1`
    * `origin_dates` - List of origin dates, one per row
    * `lags` - Number of lagged values to include

  ## Examples

      iex> known_values = %{~D[2023-01-01] => 1.0, ~D[2023-01-02] => 2.0, ~D[2023-01-03] => 3.0}
      iex> Soothsayer.AR.build_input(known_values, [~D[2023-01-02], ~D[2023-01-03]], 2)
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 2.0],
          [2.0, 3.0]
        ]
      >

  """
  @spec build_input(%{Date.t() => float()}, list(Date.t()), non_neg_integer()) :: Nx.Tensor.t()
  def build_input(known_values, origin_dates, lags) do
    rows =
      Enum.map(origin_dates, fn origin_date ->
        lagged_values =
          Enum.map((lags - 1)..0//-1, fn offset ->
            Map.get(known_values, Date.add(origin_date, -offset))
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
