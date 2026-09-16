defmodule Soothsayer.Model do
  @moduledoc """
  Defines the structure and operations for the Soothsayer forecasting model.

  ## Network layout

  Every sample the network sees is one forecast origin, laid out as
  `positions = lags + forecast_steps` timestamps: the auto-regression lags
  (oldest first) followed by the forecast targets. Without auto-regression
  a sample is a single timestamp.

  The components that depend only on the timestamp (trend, seasonality,
  events, regressors) take `{batch, positions, features}` inputs and produce
  `{batch, positions}` outputs through one shared linear layer each. Their
  sum at the lag positions is subtracted from the lags before the AR
  network, see `Soothsayer.AR`, and their values at the target positions
  are the component outputs, `{batch, forecast_steps}`, which add up to
  `combined`.
  """

  alias Soothsayer.AR
  alias Soothsayer.Events
  alias Soothsayer.LaggedRegressors
  alias Soothsayer.Layers
  alias Soothsayer.Quantiles
  alias Soothsayer.Regressors
  alias Soothsayer.Seasonality
  alias Soothsayer.Trainer
  alias Soothsayer.Trend

  defstruct [:network, :params, :config, :predict_fn]

  @typedoc """
  A model. `predict_fn` is the network's predict function compiled with
  EXLA, set by `fit/4` so that `predict/2` doesn't rebuild the network on
  every call. When it is `nil` (a model assembled by hand) `predict/2`
  builds the network eagerly instead.
  """
  @type t :: %__MODULE__{
          network: Axon.t(),
          params: term() | nil,
          config: map(),
          predict_fn: (term(), map() -> map()) | nil
        }

  @doc """
  Creates a new Soothsayer.Model struct with the given configuration.

  ## Parameters

    * `config` - A map containing the model configuration.

  ## Returns

    A new `Soothsayer.Model` struct.

  ## Examples

      iex> config = %{trend: %{enabled: true}, seasonality: %{yearly: %{enabled: true, fourier_terms: 6}}}
      iex> Soothsayer.Model.new(config)
      %Soothsayer.Model{network: ..., params: nil, config: ^config}

  """
  @spec new(map()) :: t()
  def new(config) do
    %__MODULE__{
      network: build_network(config),
      config: config
    }
  end

  @doc """
  Builds the neural network for the Soothsayer model based on the given configuration.

  ## Parameters

    * `config` - A map containing the model configuration.

  ## Returns

    An Axon neural network structure.

  ## Examples

      iex> config = %{trend: %{enabled: true}, seasonality: %{yearly: %{enabled: true, fourier_terms: 6}}}
      iex> network = Soothsayer.Model.build_network(config)
      #Axon.Node<...>

  """
  @spec build_network(map()) :: Axon.t()
  def build_network(config) do
    {combined, components} = build_network_components(config)

    outputs = %{
      combined: combined,
      trend: components.trend,
      yearly_seasonality: components.seasonality.yearly,
      weekly_seasonality: components.seasonality.weekly,
      daily_seasonality: components.seasonality.daily,
      ar: components.ar,
      events: components.events,
      regressors: components.regressors,
      lagged_regressors: components.lagged_regressors
    }

    # Only present when quantiles are configured, as a tuple in the same
    # order as config.quantiles.
    outputs =
      case components.quantiles do
        [] -> outputs
        nodes -> Map.put(outputs, :quantiles, List.to_tuple(nodes))
      end

    Axon.container(outputs)
  end

  @doc """
  Returns a display-friendly version of the network that outputs a single tensor.

  This version can be used with `Axon.Display.as_graph/2` since it doesn't use
  `Axon.container` with a map output.

  ## Examples

      iex> model = Soothsayer.new(config)
      iex> input = %{"trend" => Nx.template({1, 1, 1}, :f32), ...}
      iex> Axon.Display.as_graph(Soothsayer.Model.display_network(model.config), input)

  """
  @spec display_network(map()) :: Axon.t()
  def display_network(config) do
    {combined, _components} = build_network_components(config)
    combined
  end

  defp build_network_components(config) do
    lags = AR.lags(config)
    positions = AR.positions(config)

    # Trend
    trend_input = Trend.build_input(config)
    trend = Trend.build_component(trend_input, config)

    # Seasonality, one component per period. Periods missing from the config
    # (only in unit tests, Soothsayer.new fills them all in) become zero.
    seasonality_inputs = Seasonality.build_inputs(config)

    seasonality =
      seasonality_inputs
      |> Seasonality.build_components(config)
      |> apply_seasonality_mode(trend, config)
      |> then(fn components ->
        Map.new(Seasonality.periods(), &{&1, Map.get(components, &1, Axon.constant(0))})
      end)

    # Events
    events_input = Events.build_network_input(config)
    events = Events.build_component(events_input, config)

    # Future regressors
    regressors_input = Regressors.build_network_input(config)
    regressors = Regressors.build_component(regressors_input, config)

    # Everything that depends only on the timestamp, over all positions.
    # Its values at the lag positions are what the AR network subtracts
    # from the lags, its values at the target positions are the forecast.
    nonstationary =
      Axon.add(
        [trend] ++ Enum.map(Seasonality.periods(), &seasonality[&1]) ++ [events, regressors],
        name: "nonstationary"
      )

    nonstationary_at_lags =
      if lags > 0 do
        Layers.slice_positions(nonstationary, 0..(lags - 1), "nonstationary_at_lags")
      else
        Axon.constant(0)
      end

    # AR
    ar_input = AR.build_network_input(config)
    ar = AR.build_component(ar_input, nonstationary_at_lags, config)

    # Lagged regressors
    lagged_regressors_input = LaggedRegressors.build_network_input(config)
    lagged_regressors = LaggedRegressors.build_component(lagged_regressors_input, config)

    target_range = lags..(positions - 1)
    trend_at_targets = Layers.slice_positions(trend, target_range, "trend_at_targets")

    seasonality_at_targets =
      Map.new(seasonality, fn {period, component} ->
        {period, Layers.slice_positions(component, target_range, "#{period}_at_targets")}
      end)

    events_at_targets = Layers.slice_positions(events, target_range, "events_at_targets")

    regressors_at_targets =
      Layers.slice_positions(regressors, target_range, "regressors_at_targets")

    combined =
      Axon.add(
        [trend_at_targets] ++
          Enum.map(Seasonality.periods(), &seasonality_at_targets[&1]) ++
          [ar, events_at_targets, regressors_at_targets, lagged_regressors]
      )

    # Quantile heads see every input the components see
    inputs =
      Enum.reject(
        [trend_input] ++
          Enum.map(Seasonality.periods(), &seasonality_inputs[&1]) ++
          [ar_input, events_input, regressors_input, lagged_regressors_input],
        &is_nil/1
      )

    quantiles = Quantiles.build_components(inputs, combined, config)

    {combined,
     %{
       trend: trend_at_targets,
       seasonality: seasonality_at_targets,
       ar: ar,
       events: events_at_targets,
       regressors: regressors_at_targets,
       lagged_regressors: lagged_regressors,
       quantiles: quantiles
     }}
  end

  # Multiplicative seasonality scales the seasonal effect by the trend. The
  # network works in normalized y space, where the trend is centered near
  # zero, so the multiplier is the trend plus the series level (mean / std).
  # The level is only known after fit computes the normalization, which is
  # why fit rebuilds the network; before that the level is zero.
  defp apply_seasonality_mode(
         seasonality,
         trend,
         %{seasonality: %{mode: :multiplicative}} = config
       ) do
    scale =
      Axon.add(detach_at_lags(trend, AR.lags(config)), Axon.constant(series_level(config)))

    # A disabled period is a scalar constant and stays one; multiplying it by
    # the scale would turn it into a full tensor of zeros that predict would
    # then report as a component.
    Map.new(seasonality, fn {period, component} ->
      if Seasonality.enabled?(config, period),
        do: {period, Axon.multiply(component, scale)},
        else: {period, component}
    end)
  end

  defp apply_seasonality_mode(seasonality, _trend, _config), do: seasonality

  # At the lag positions the seasonal terms are only there to be subtracted
  # from the lags, and NeuralProphet detaches the trend inside them so that
  # subtraction doesn't train the trend a second time through the lags.
  defp detach_at_lags(trend, 0), do: trend

  defp detach_at_lags(trend, lags) do
    Axon.nx(
      trend,
      fn tensor ->
        if Nx.rank(tensor) == 0 do
          tensor
        else
          Nx.concatenate(
            [
              Nx.Defn.Kernel.stop_grad(tensor[[.., 0..(lags - 1)]]),
              tensor[[.., lags..(Nx.axis_size(tensor, 1) - 1)]]
            ],
            axis: 1
          )
        end
      end,
      name: "trend_detached_at_lags"
    )
  end

  defp series_level(config) do
    case get_in(config, [:normalization, :y]) do
      %{mean: mean, std: std} -> scalar(mean) / scalar(std)
      nil -> 0.0
    end
  end

  defp scalar(tensor), do: tensor |> Nx.squeeze() |> Nx.to_number()

  @doc """
  Fits the Soothsayer model to the provided data.

  ## Parameters

    * `model` - A `Soothsayer.Model` struct.
    * `x` - A map of input tensors.
    * `y` - A tensor of target values, `{samples, forecast_steps}`.
    * `epochs` - The number of training epochs.

  ## Returns

    An updated `Soothsayer.Model` struct with fitted parameters.

  ## Examples

      iex> model = Soothsayer.Model.new(config)
      iex> x = %{"trend" => trend_tensor, "yearly" => yearly_tensor, "weekly" => weekly_tensor}
      iex> y = target_tensor
      iex> fitted_model = Soothsayer.Model.fit(model, x, y, 100)
      %Soothsayer.Model{...}

  """
  @spec fit(t(), %{String.t() => Nx.Tensor.t()}, Nx.Tensor.t(), non_neg_integer()) :: t()
  def fit(model, x, y, epochs) do
    trained_params = Trainer.fit(model.network, x, y, epochs, model.config)
    {_init_fn, predict_fn} = Axon.build(model.network, compiler: EXLA)
    %{model | params: trained_params, predict_fn: predict_fn}
  end

  @doc """
  Makes predictions using a fitted Soothsayer model.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct.
    * `x` - A map of input tensors.

  ## Returns

    A map containing the predicted values for each component and the
    combined prediction, each `{samples, forecast_steps}`.

  ## Examples

      iex> fitted_model = Soothsayer.Model.fit(model, training_x, training_y, 100)
      iex> x = %{"trend" => future_trend_tensor, "yearly" => future_yearly_tensor, "weekly" => future_weekly_tensor, "daily" => future_daily_tensor}
      iex> predictions = Soothsayer.Model.predict(fitted_model, x)
      %{
        combined: #Nx.Tensor<...>,
        trend: #Nx.Tensor<...>,
        yearly_seasonality: #Nx.Tensor<...>,
        weekly_seasonality: #Nx.Tensor<...>,
        daily_seasonality: #Nx.Tensor<...>
      }

  """
  @spec predict(t(), %{String.t() => Nx.Tensor.t()}) :: %{
          optional(:quantiles) => tuple(),
          combined: Nx.Tensor.t(),
          trend: Nx.Tensor.t(),
          yearly_seasonality: Nx.Tensor.t(),
          weekly_seasonality: Nx.Tensor.t(),
          daily_seasonality: Nx.Tensor.t(),
          ar: Nx.Tensor.t(),
          events: Nx.Tensor.t(),
          regressors: Nx.Tensor.t(),
          lagged_regressors: Nx.Tensor.t()
        }
  def predict(%{predict_fn: nil} = model, x) do
    {_init_fn, predict_fn} = Axon.build(model.network)
    predict_fn.(model.params, x)
  end

  def predict(model, x), do: model.predict_fn.(model.params, x)
end
