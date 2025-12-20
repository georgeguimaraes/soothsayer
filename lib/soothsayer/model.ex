defmodule Soothsayer.Model do
  @moduledoc """
  Defines the structure and operations for the Soothsayer forecasting model.
  """

  alias Soothsayer.AR
  alias Soothsayer.Events
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

    Axon.container(%{
      combined: combined,
      trend: components.trend,
      yearly_seasonality: components.yearly,
      weekly_seasonality: components.weekly,
      ar: components.ar,
      events: components.events
    })
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
    seasonality = Seasonality.build_components(seasonality_inputs, config)

    # AR
    ar_input = AR.build_network_input(config)
    ar_component = AR.build_component(ar_input, config)

    # Events
    events_input = Events.build_network_input(%{events: config[:events] || %{}})
    events_component = Events.build_component(events_input, %{events: config[:events] || %{}})

    combined =
      Axon.add([trend, seasonality.yearly, seasonality.weekly, ar_component, events_component])

    {combined,
     %{
       trend: trend,
       yearly: seasonality.yearly,
       weekly: seasonality.weekly,
       ar: ar_component,
       events: events_component
     }}
  end

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
          combined: Nx.Tensor.t(),
          trend: Nx.Tensor.t(),
          yearly_seasonality: Nx.Tensor.t(),
          weekly_seasonality: Nx.Tensor.t()
        }
  def predict(model, x) do
    {_init_fn, predict_fn} = Axon.build(model.network)
    predict_fn.(model.params, x)
  end
end
