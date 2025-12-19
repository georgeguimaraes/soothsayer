defmodule Soothsayer.Model do
  @moduledoc """
  Defines the structure and operations for the Soothsayer forecasting model.
  """

  alias Soothsayer.Trainer

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
    n_changepoints = get_in(config, [:trend, :n_changepoints]) || 0
    trend_input = Axon.input("trend", shape: {nil, 1 + n_changepoints})
    yearly_input = Axon.input("yearly", shape: {nil, 2 * config.seasonality.yearly.fourier_terms})
    weekly_input = Axon.input("weekly", shape: {nil, 2 * config.seasonality.weekly.fourier_terms})

    trend =
      if config.trend.enabled do
        Axon.dense(trend_input, 1, activation: :linear, name: "trend_dense")
      else
        Axon.constant(0)
      end

    yearly_seasonality =
      if config.seasonality.yearly.enabled do
        Axon.dense(yearly_input, 1, activation: :linear)
      else
        Axon.constant(0)
      end

    weekly_seasonality =
      if config.seasonality.weekly.enabled do
        Axon.dense(weekly_input, 1, activation: :linear)
      else
        Axon.constant(0)
      end

    {ar_component, _ar_input} =
      if config[:ar][:enabled] do
        ar_input = Axon.input("ar", shape: {nil, config.ar.n_lags})
        ar_layers = Map.get(config.ar, :layers, [])

        ar =
          case ar_layers do
            [] ->
              # Linear AR (current behavior)
              Axon.dense(ar_input, 1, activation: :linear, name: "ar_dense_out")

            layers ->
              # Deep AR-Net with hidden layers
              {hidden, _idx} =
                Enum.reduce(layers, {ar_input, 0}, fn units, {acc, idx} ->
                  {Axon.dense(acc, units, activation: :relu, name: "ar_dense_#{idx}"), idx + 1}
                end)

              Axon.dense(hidden, 1, activation: :linear, name: "ar_dense_out")
          end

        {ar, ar_input}
      else
        {Axon.constant(0), nil}
      end

    combined = Axon.add([trend, yearly_seasonality, weekly_seasonality, ar_component])

    container = %{
      combined: combined,
      trend: trend,
      yearly_seasonality: yearly_seasonality,
      weekly_seasonality: weekly_seasonality,
      ar: ar_component
    }

    Axon.container(container)
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
