defmodule Soothsayer.Model do
  @moduledoc """
  Defines the structure and operations for the Soothsayer forecasting model.
  """

  alias Soothsayer.AR
  alias Soothsayer.Events
  alias Soothsayer.Quantiles
  alias Soothsayer.Regressors
  alias Soothsayer.Seasonality
  alias Soothsayer.Trainer
  alias Soothsayer.Trend

  defstruct [:network, :params, :config]

  @type t :: %__MODULE__{
          network: Axon.t(),
          params: term() | nil,
          config: map()
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
      yearly_seasonality: components.yearly,
      weekly_seasonality: components.weekly,
      ar: components.ar,
      events: components.events,
      regressors: components.regressors
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
      iex> input = %{"trend" => Nx.template({1, 1}, :f32), ...}
      iex> Axon.Display.as_graph(Soothsayer.Model.display_network(model.config), input)

  """
  @spec display_network(map()) :: Axon.t()
  def display_network(config) do
    {combined, _components} = build_network_components(config)
    combined
  end

  defp build_network_components(config) do
    # Trend
    trend_input = Trend.build_input(config)
    trend = Trend.build_component(trend_input, config)

    # Seasonality
    seasonality_inputs = Seasonality.build_inputs(config)

    seasonality =
      seasonality_inputs
      |> Seasonality.build_components(config)
      |> apply_seasonality_mode(trend, config)

    # AR
    ar_input = AR.build_network_input(config)
    step_mask_input = AR.build_step_mask_input(config)
    ar_component = AR.build_component(ar_input, step_mask_input, config)

    # Events
    events_input = Events.build_network_input(%{events: config[:events] || %{}})
    events_component = Events.build_component(events_input, %{events: config[:events] || %{}})

    # Future regressors
    regressors_input = Regressors.build_network_input(config)
    regressors_component = Regressors.build_component(regressors_input, config)

    combined =
      Axon.add([
        trend,
        seasonality.yearly,
        seasonality.weekly,
        ar_component,
        events_component,
        regressors_component
      ])

    # Quantile heads see every input the components see
    inputs =
      Enum.reject(
        [
          trend_input,
          seasonality_inputs.yearly,
          seasonality_inputs.weekly,
          ar_input,
          step_mask_input,
          events_input,
          regressors_input
        ],
        &is_nil/1
      )

    quantiles = Quantiles.build_components(inputs, combined, config)

    {combined,
     %{
       trend: trend,
       yearly: seasonality.yearly,
       weekly: seasonality.weekly,
       ar: ar_component,
       events: events_component,
       regressors: regressors_component,
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
    scale = Axon.add(trend, Axon.constant(series_level(config)))

    %{
      yearly: Axon.multiply(seasonality.yearly, scale),
      weekly: Axon.multiply(seasonality.weekly, scale)
    }
  end

  defp apply_seasonality_mode(seasonality, _trend, _config), do: seasonality

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
    * `y` - A tensor of target values.
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
    %{model | params: trained_params}
  end

  @doc """
  Makes predictions using a fitted Soothsayer model.

  ## Parameters

    * `model` - A fitted `Soothsayer.Model` struct.
    * `x` - A map of input tensors.

  ## Returns

    A map containing the predicted values for each component and the combined prediction.

  ## Examples

      iex> fitted_model = Soothsayer.Model.fit(model, training_x, training_y, 100)
      iex> x = %{"trend" => future_trend_tensor, "yearly" => future_yearly_tensor, "weekly" => future_weekly_tensor}
      iex> predictions = Soothsayer.Model.predict(fitted_model, x)
      %{
        combined: #Nx.Tensor<...>,
        trend: #Nx.Tensor<...>,
        yearly_seasonality: #Nx.Tensor<...>,
        weekly_seasonality: #Nx.Tensor<...>
      }

  """
  @spec predict(t(), %{String.t() => Nx.Tensor.t()}) :: %{
          optional(:quantiles) => tuple(),
          combined: Nx.Tensor.t(),
          trend: Nx.Tensor.t(),
          yearly_seasonality: Nx.Tensor.t(),
          weekly_seasonality: Nx.Tensor.t(),
          ar: Nx.Tensor.t(),
          events: Nx.Tensor.t(),
          regressors: Nx.Tensor.t()
        }
  def predict(model, x) do
    {_init_fn, predict_fn} = Axon.build(model.network)
    predict_fn.(model.params, x)
  end
end
