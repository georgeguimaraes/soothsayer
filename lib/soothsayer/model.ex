defmodule Soothsayer.Model do
  @moduledoc """
  Defines the structure and operations for the Soothsayer forecasting model.
  """

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
    trend_input = Axon.input("trend", shape: {nil, 1})
    yearly_input = Axon.input("yearly", shape: {nil, 2 * config.seasonality.yearly.fourier_terms})
    weekly_input = Axon.input("weekly", shape: {nil, 2 * config.seasonality.weekly.fourier_terms})

    trend =
      if config.trend.enabled do
        Axon.dense(trend_input, 1, activation: :linear)
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
    {init_fn, _predict_fn} = Axon.build(model.network)
    initial_params = init_fn.(x, Axon.ModelState.empty())

    ar_reg = get_in(model.config, [:ar, :regularization])

    trained_params =
      if ar_reg do
        # Custom training with L1 regularization on AR weights
        train_with_regularization(model, x, y, epochs, initial_params, ar_reg)
      else
        # Standard training without regularization
        model.network
        |> Axon.Loop.trainer(
          &Axon.Losses.huber(&1, &2.combined, reduction: :mean),
          Polaris.Optimizers.adam(learning_rate: model.config.learning_rate)
        )
        |> Axon.Loop.run(Stream.repeatedly(fn -> {x, y} end), initial_params,
          epochs: epochs,
          iterations: elem(Nx.shape(y), 0),
          compiler: EXLA
        )
      end

    %{model | params: trained_params}
  end

  @doc false
  def train_with_regularization_public(model, x, y, epochs, initial_params, ar_reg) do
    train_with_regularization(model, x, y, epochs, initial_params, ar_reg)
  end

  defp train_with_regularization(model, x, y, epochs, initial_params, ar_reg) do
    # Find AR layer names at runtime
    ar_layer_names =
      initial_params.data
      |> Map.keys()
      |> Enum.filter(&String.starts_with?(&1, "ar_dense"))

    {_init_fn, predict_fn} = Axon.build(model.network)
    {init_optim_fn, update_fn} = Polaris.Optimizers.adam(learning_rate: model.config.learning_rate)

    # Define the objective function with L1 penalty
    objective_fn = fn params, x_input, y_target ->
      predictions = predict_fn.(params, x_input)
      base_loss = Axon.Losses.huber(y_target, predictions.combined, reduction: :mean)
      ar_penalty = compute_ar_l1_penalty_jit(params, ar_layer_names)
      Nx.add(base_loss, Nx.multiply(ar_reg, ar_penalty))
    end

    # JIT compile the train step
    train_step_fn = fn params, opt_state, x_input, y_target ->
      {loss, grads} = Nx.Defn.value_and_grad(params, fn p -> objective_fn.(p, x_input, y_target) end)
      {updates, new_opt_state} = update_fn.(grads, opt_state, params)
      new_params = Polaris.Updates.apply_updates(params, updates)
      {loss, new_params, new_opt_state}
    end

    jit_train_step = EXLA.jit(train_step_fn)

    initial_opt_state = init_optim_fn.(initial_params)
    iterations = elem(Nx.shape(y), 0)

    log_interval = max(div(epochs, 10), 1)

    {final_params, _final_opt_state} =
      Enum.reduce(1..epochs, {initial_params, initial_opt_state}, fn epoch, {params, opt_state} ->
        {final_loss, new_params, new_opt_state} =
          Enum.reduce(1..iterations, {nil, params, opt_state}, fn _iter, {_loss, p, os} ->
            jit_train_step.(p, os, x, y)
          end)

        if rem(epoch - 1, log_interval) == 0 do
          IO.puts("Epoch: #{epoch - 1}, loss: #{Nx.to_number(final_loss)}")
        end

        {new_params, new_opt_state}
      end)

    final_params
  end

  defp compute_ar_l1_penalty_jit(params, ar_layer_names) do
    if Enum.empty?(ar_layer_names) do
      Nx.tensor(0.0)
    else
      ar_layer_names
      |> Enum.map(fn layer_name ->
        kernel = params.data[layer_name]["kernel"]
        Nx.sum(Nx.abs(kernel))
      end)
      |> Enum.reduce(&Nx.add/2)
    end
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
