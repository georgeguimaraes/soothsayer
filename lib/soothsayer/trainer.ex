defmodule Soothsayer.Trainer do
  @moduledoc """
  Training functionality for Soothsayer models.

  Handles standard training and custom training with L1 regularization
  on specified layer weights (AR, trend, etc.).
  """

  @doc """
  Trains a network on the provided data.

  ## Parameters

    * `network` - An Axon neural network.
    * `x` - A map of input tensors.
    * `y` - A tensor of target values.
    * `epochs` - The number of training epochs.
    * `config` - A map containing training configuration:
      - `:learning_rate` - Learning rate for the optimizer.
      - `:ar` - Optional map with `:regularization` for AR L1 penalty.
      - `:trend` - Optional map with `:regularization` for trend L1 penalty.

  ## Returns

    The trained `Axon.ModelState`.

  ## Examples

      iex> config = %{learning_rate: 0.1}
      iex> params = Soothsayer.Trainer.fit(network, x, y, 100, config)
      %Axon.ModelState{...}

  """
  @spec fit(Axon.t(), %{String.t() => Nx.Tensor.t()}, Nx.Tensor.t(), non_neg_integer(), map()) ::
          %Axon.ModelState{}
  def fit(network, x, y, epochs, config) do
    {init_fn, _predict_fn} = Axon.build(network)
    initial_params = init_fn.(x, Axon.ModelState.empty())

    ar_reg = get_in(config, [:ar, :regularization])
    trend_reg = get_in(config, [:trend, :regularization])

    if ar_reg || trend_reg do
      train_with_regularization(network, x, y, epochs, initial_params, config)
    else
      train_standard(network, x, y, epochs, initial_params, config)
    end
  end

  @doc """
  Computes L1 penalty for specified layer kernels.

  ## Parameters

    * `params` - An `Axon.ModelState` containing the model parameters.
    * `layer_names` - A list of layer names to include in the penalty.

  ## Returns

    A scalar tensor with the sum of absolute values of all kernel weights
    in the specified layers.

  ## Examples

      iex> penalty = Soothsayer.Trainer.compute_l1_penalty(params, ["ar_dense_out"])
      #Nx.Tensor<f32 6.0>

  """
  @spec compute_l1_penalty(%Axon.ModelState{}, list(String.t())) :: Nx.Tensor.t()
  def compute_l1_penalty(params, layer_names) do
    if Enum.empty?(layer_names) do
      Nx.tensor(0.0)
    else
      layer_names
      |> Enum.map(fn layer_name ->
        kernel = params.data[layer_name]["kernel"]
        Nx.sum(Nx.abs(kernel))
      end)
      |> Enum.reduce(&Nx.add/2)
    end
  end

  defp train_standard(network, x, y, epochs, initial_params, config) do
    network
    |> Axon.Loop.trainer(
      &Axon.Losses.huber(&1, &2.combined, reduction: :mean),
      Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    )
    |> Axon.Loop.run(Stream.repeatedly(fn -> {x, y} end), initial_params,
      epochs: epochs,
      iterations: elem(Nx.shape(y), 0),
      compiler: EXLA
    )
  end

  defp train_with_regularization(network, x, y, epochs, initial_params, config) do
    ar_reg = get_in(config, [:ar, :regularization]) || 0.0
    trend_reg = get_in(config, [:trend, :regularization]) || 0.0

    regularization_layers = build_regularization_layers(initial_params, ar_reg, trend_reg)

    {_init_fn, predict_fn} = Axon.build(network)
    {init_optim_fn, update_fn} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)

    objective_fn = build_objective_fn(predict_fn, regularization_layers)
    train_step_fn = build_train_step_fn(objective_fn, update_fn)
    jit_train_step = EXLA.jit(train_step_fn)

    initial_opt_state = init_optim_fn.(initial_params)
    iterations = elem(Nx.shape(y), 0)

    run_training_loop(epochs, iterations, initial_params, initial_opt_state, x, y, jit_train_step)
  end

  defp build_regularization_layers(params, ar_reg, trend_reg) do
    ar_layers =
      if ar_reg > 0 do
        params.data
        |> Map.keys()
        |> Enum.filter(&String.starts_with?(&1, "ar_dense"))
        |> Enum.map(fn name -> {name, ar_reg} end)
      else
        []
      end

    trend_layers =
      if trend_reg > 0 do
        params.data
        |> Map.keys()
        |> Enum.filter(&String.starts_with?(&1, "trend_dense"))
        |> Enum.map(fn name -> {name, trend_reg} end)
      else
        []
      end

    ar_layers ++ trend_layers
  end

  defp build_objective_fn(predict_fn, regularization_layers) do
    fn params, x_input, y_target ->
      predictions = predict_fn.(params, x_input)
      base_loss = Axon.Losses.huber(y_target, predictions.combined, reduction: :mean)
      penalty = compute_weighted_l1_penalty(params, regularization_layers)
      Nx.add(base_loss, penalty)
    end
  end

  defp compute_weighted_l1_penalty(params, regularization_layers) do
    if Enum.empty?(regularization_layers) do
      Nx.tensor(0.0)
    else
      regularization_layers
      |> Enum.map(fn {layer_name, reg_weight} ->
        kernel = params.data[layer_name]["kernel"]
        Nx.multiply(reg_weight, Nx.sum(Nx.abs(kernel)))
      end)
      |> Enum.reduce(&Nx.add/2)
    end
  end

  defp build_train_step_fn(objective_fn, update_fn) do
    fn params, opt_state, x_input, y_target ->
      {loss, grads} = Nx.Defn.value_and_grad(params, fn p -> objective_fn.(p, x_input, y_target) end)
      {updates, new_opt_state} = update_fn.(grads, opt_state, params)
      new_params = Polaris.Updates.apply_updates(params, updates)
      {loss, new_params, new_opt_state}
    end
  end

  defp run_training_loop(epochs, iterations, initial_params, initial_opt_state, x, y, jit_train_step) do
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
end
