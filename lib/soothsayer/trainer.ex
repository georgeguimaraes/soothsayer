defmodule Soothsayer.Trainer do
  @moduledoc """
  Training functionality for Soothsayer models.

  Handles standard training and custom training with L1 regularization
  on specified layer weights (AR, trend, etc.).

  Training runs in shuffled minibatches of samples, one sample being a
  forecast origin with its `forecast_steps` targets (a single timestamp
  without auto-regression). One epoch is one pass over the samples, and the
  batch size comes from `config[:batch_size]`, or is derived from the number
  of samples when that is `nil`.
  """

  import Nx.Defn

  alias Soothsayer.Events
  alias Soothsayer.Quantiles
  alias Soothsayer.Regressors
  alias Soothsayer.Seasonality

  @min_auto_batch_size 16
  @max_auto_batch_size 512
  @min_auto_epochs 20
  @max_auto_epochs 500

  # Learning rate range test bounds, the same as NeuralProphet's
  @range_test_min_learning_rate 1.0e-6
  @range_test_max_learning_rate 10.0
  @range_test_skip_begin 10
  @range_test_skip_end 3
  @range_test_smoothing_half_window 5
  @range_test_divergence_factor 4.0

  @adamw_weight_decay 1.0e-3

  @doc """
  Trains a network on the provided data.

  ## Parameters

    * `network` - An Axon neural network.
    * `x` - A map of input tensors, one sample per row.
    * `y` - A tensor of target values, `{samples, forecast_steps}`.
    * `epochs` - The number of training epochs.
    * `config` - A map containing training configuration:
      - `:learning_rate` - Learning rate for the optimizer.
      - `:batch_size` - Samples per gradient step. `nil` (or missing) picks
        a size based on the number of samples, see `auto_batch_size/1`.
      - `:seed` - Integer seed for parameter initialization and batch
        shuffling, so two fits with the same seed produce the same model.
        `nil` (or missing) leaves both random. Seeding the shuffle reseeds
        `:rand` in the calling process.
      - `:schedule` - `:constant` (default) keeps the learning rate fixed,
        `:one_cycle` runs NeuralProphet's three-phase one-cycle schedule
        peaking at the learning rate, see `one_cycle/3`.
      - `:optimizer` - `:adam` (default) or `:adamw` with weight decay
        `1.0e-3`, NeuralProphet's default optimizer.

    `learning_rate` and `epochs` must be concrete numbers here. Resolve
    `:auto` with `resolve_learning_rate/4` and `auto_epochs/1` first.
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
          Soothsayer.Model.model_state()
  def fit(network, x, y, epochs, config) do
    seed = config[:seed]
    if seed, do: :rand.seed(:exsss, {seed, seed, seed})

    {init_fn, _predict_fn} = Axon.build(network, build_options(seed))
    initial_params = init_fn.(x, Axon.ModelState.empty())

    n_rows = Nx.axis_size(y, 0)
    batch_size = min(config[:batch_size] || auto_batch_size(n_rows), n_rows)

    unless is_number(config.learning_rate) and is_integer(epochs) do
      raise ArgumentError,
            "Trainer.fit needs a numeric learning_rate and integer epochs, got " <>
              "#{inspect(config.learning_rate)} and #{inspect(epochs)}. " <>
              "Resolve :auto with resolve_learning_rate/4 and auto_epochs/1 first."
    end

    total_steps = epochs * div(n_rows, batch_size)
    optimizer = build_optimizer(config, config.learning_rate, total_steps)

    # Axon.Loop calls the loss with targets and predictions only, so a
    # penalty or a per-sample weight needs the custom loop.
    if regularization_terms(initial_params, config) == [] and
         local_terms(initial_params, config) == [] and
         not Map.has_key?(x, "sample_weight") do
      train_standard(network, x, y, epochs, batch_size, initial_params, config, optimizer)
    else
      train_custom(network, x, y, epochs, batch_size, initial_params, config, optimizer)
    end
  end

  @doc """
  Per-sample loss weights that favour recent rows, NeuralProphet's "newer
  samples weight".

  `target_times` holds the span-normalized time of every target position,
  0 at the first training timestamp and 1 at the last. Rows at or before
  `start` (a fraction of the span) get weight `1 / weight`, the last row
  gets 1, and in between the weight follows a half cosine, so the ramp is
  smooth rather than a cutoff. The mean loss shrinks by up to `1 / weight`,
  which only rescales the gradient.

  ## Examples

      iex> times = Nx.tensor([[0.0], [0.5], [1.0]])
      iex> Soothsayer.Trainer.recency_weights(times, %{weight: 2, start: 0.0}) |> Nx.to_flat_list()
      [0.5, 0.75, 1.0]

  """
  @spec recency_weights(Nx.Tensor.t(), %{weight: number(), start: number()}) :: Nx.Tensor.t()
  def recency_weights(target_times, %{weight: weight, start: start}) do
    progress =
      target_times
      |> Nx.subtract(start)
      |> Nx.divide(1.0 - start)
      |> Nx.clip(0.0, 1.0)

    ramp =
      progress
      |> Nx.subtract(1.0)
      |> Nx.multiply(:math.pi())
      |> Nx.cos()
      |> Nx.multiply(0.5)
      |> Nx.add(0.5)

    ramp
    |> Nx.multiply(weight - 1)
    |> Nx.add(1.0)
    |> Nx.divide(weight)
  end

  @doc """
  Picks the number of epochs from the number of training rows.

  NeuralProphet's heuristic: `10 * ceil(100 / n * 2 ** (2.25 * log10(10 + n)))`,
  clamped to 20..500. Small datasets get many passes, large ones fewer.

  ## Examples

      iex> Soothsayer.Trainer.auto_epochs(130)
      220
      iex> Soothsayer.Trainer.auto_epochs(2615)
      80

  """
  @spec auto_epochs(pos_integer()) :: pos_integer()
  def auto_epochs(n_rows) when n_rows > 0 do
    epochs = 10 * ceil(100 / n_rows * :math.pow(2, 2.25 * :math.log10(10 + n_rows)))

    epochs
    |> max(@min_auto_epochs)
    |> min(@max_auto_epochs)
  end

  @doc """
  Resolves `config.learning_rate`, running the range test when it is `:auto`.

  Returns the learning rate as a float.
  """
  @spec resolve_learning_rate(Axon.t(), %{String.t() => Nx.Tensor.t()}, Nx.Tensor.t(), map()) ::
          float()
  def resolve_learning_rate(network, x, y, %{learning_rate: :auto} = config) do
    find_learning_rate(network, x, y, config)
  end

  def resolve_learning_rate(_network, _x, _y, %{learning_rate: learning_rate}) do
    learning_rate
  end

  @doc """
  Learning rate range test, as NeuralProphet runs it through Lightning.

  Trains fresh parameters for about a hundred minibatch steps while the
  learning rate climbs exponentially from `1.0e-6` to `10`, records the
  loss after every step, smooths the curve with a Hamming window and returns
  the learning rate at its steepest descent, skipping the first 10 and last
  3 points. The number of steps grows slowly with the size of the real
  training run: `100 + 30 * log10(1 + total_steps / 1000)`.

  The suggested rate is where the loss is falling fastest, which is a
  good peak for a one-cycle schedule and a reasonable constant rate.
  """
  @spec find_learning_rate(Axon.t(), %{String.t() => Nx.Tensor.t()}, Nx.Tensor.t(), map()) ::
          float()
  def find_learning_rate(network, x, y, config) do
    seed = config[:seed]
    if seed, do: :rand.seed(:exsss, {seed, seed, seed})

    {init_fn, predict_fn} = Axon.build(network, build_options(seed))
    initial_params = init_fn.(x, Axon.ModelState.empty())

    n_rows = Nx.axis_size(y, 0)
    batch_size = min(config[:batch_size] || auto_batch_size(n_rows), n_rows)
    epochs = if is_integer(config.epochs), do: config.epochs, else: auto_epochs(n_rows)
    main_total_steps = epochs * div(n_rows, batch_size)
    num_training = 100 + trunc(:math.log10(1 + main_total_steps / 1000) * 30)

    learning_rates = range_test_learning_rates(num_training)

    schedule = fn count ->
      ratio = @range_test_max_learning_rate / @range_test_min_learning_rate
      Nx.multiply(@range_test_min_learning_rate, Nx.pow(ratio, Nx.divide(count, num_training)))
    end

    {init_optimizer_fn, update_fn} = Polaris.Optimizers.adam(learning_rate: schedule)
    quantiles = config[:quantiles] || []

    # No penalty during the range test, so the weights map stays empty. The
    # sample weights do apply, as in NeuralProphet, so the curve is measured
    # on the loss that training will minimize.
    objective_fn = fn params, x_input, y_target, _weights ->
      loss(y_target, predict_fn.(params, x_input), quantiles, Map.get(x_input, "sample_weight"))
    end

    jit_train_step = EXLA.jit(build_train_step_fn(objective_fn, update_fn))

    train_step = fn params, opt_state, x_batch, y_batch ->
      jit_train_step.(params, opt_state, x_batch, y_batch, {%{}, %{}})
    end

    # Each step updates on one minibatch like real training, but the loss
    # that goes on the curve is measured on the whole training set, since a
    # single minibatch loss is too noisy to find the steepest descent.
    full_loss = EXLA.jit(objective_fn)

    {losses, _params, _opt_state} =
      x
      |> batches(y, batch_size)
      |> Stream.cycle()
      |> Enum.take(num_training)
      |> Enum.reduce({[], initial_params, init_optimizer_fn.(initial_params)}, fn
        {x_batch, y_batch}, {losses, params, opt_state} ->
          {_batch_loss, params, opt_state} = train_step.(params, opt_state, x_batch, y_batch)
          {[Nx.to_number(full_loss.(params, x, y, {%{}, %{}})) | losses], params, opt_state}
      end)

    suggest_learning_rate(Enum.reverse(losses), learning_rates)
  end

  defp range_test_learning_rates(num_training) do
    ratio = @range_test_max_learning_rate / @range_test_min_learning_rate

    Enum.map(0..(num_training - 1), fn step ->
      @range_test_min_learning_rate * :math.pow(ratio, step / num_training)
    end)
  end

  @doc """
  Picks the learning rate at the steepest descent of a smoothed loss curve.

  The curve is cut where the run diverges: the first loss above four times
  the best loss so far (Lightning's early stop threshold for the range
  test) or a non-finite loss. Past that point the losses are huge and
  bounce around, and a bounce downwards would look like the steepest
  descent of the whole curve. Exposed for testing.
  """
  @spec suggest_learning_rate(list(number()), list(float())) :: float()
  def suggest_learning_rate(losses, learning_rates) do
    losses = truncate_at_divergence(losses)
    smoothed = hamming_smooth(losses, @range_test_smoothing_half_window)
    gradient = central_gradient(smoothed)

    candidates =
      gradient
      |> Enum.with_index()
      |> Enum.drop(@range_test_skip_begin)
      |> Enum.drop(-@range_test_skip_end)

    {_steepest, index} = Enum.min_by(candidates, fn {slope, _index} -> slope end)
    Enum.at(learning_rates, index)
  end

  # Nx.to_number returns :nan, :infinity or :neg_infinity for non-finite values
  defp finite?(value), do: is_number(value)

  defp keep_until_divergence(loss, {kept, best}) do
    if diverged?(loss, best),
      do: {:halt, {kept, best}},
      else: {:cont, {[loss | kept], min(best || loss, loss)}}
  end

  defp diverged?(loss, best) do
    not finite?(loss) or (best != nil and loss > @range_test_divergence_factor * best)
  end

  defp truncate_at_divergence(losses) do
    {kept, _best} = Enum.reduce_while(losses, {[], nil}, &keep_until_divergence/2)

    # A curve that diverges right away leaves nothing to pick from, so it
    # is used whole, non-finite values capped at the largest finite one.
    minimum_points = @range_test_skip_begin + @range_test_skip_end + 2

    if length(kept) >= minimum_points, do: Enum.reverse(kept), else: cap_non_finite(losses)
  end

  defp cap_non_finite(losses) do
    finite = Enum.filter(losses, &finite?/1)
    ceiling = if finite == [], do: 0.0, else: Enum.max(finite)
    Enum.map(losses, fn loss -> if finite?(loss), do: loss, else: ceiling end)
  end

  defp hamming_smooth(values, half_window) do
    window_size = 2 * half_window

    weights =
      Enum.map(0..(window_size - 1), fn i ->
        0.54 - 0.46 * :math.cos(2 * :math.pi() * i / (window_size - 1))
      end)

    weight_sum = Enum.sum(weights)

    padded =
      List.duplicate(hd(values), half_window) ++
        values ++ List.duplicate(List.last(values), half_window)

    padded
    |> Enum.chunk_every(window_size, 1, :discard)
    |> Enum.take(length(values))
    |> Enum.map(fn window ->
      Enum.zip_with(window, weights, &(&1 * &2)) |> Enum.sum() |> Kernel./(weight_sum)
    end)
  end

  defp central_gradient(values) do
    count = length(values)
    indexed = List.to_tuple(values)

    Enum.map(0..(count - 1), fn i ->
      cond do
        count == 1 -> 0.0
        i == 0 -> elem(indexed, 1) - elem(indexed, 0)
        i == count - 1 -> elem(indexed, i) - elem(indexed, i - 1)
        true -> (elem(indexed, i + 1) - elem(indexed, i - 1)) / 2
      end
    end)
  end

  @doc """
  NeuralProphet's three-phase one-cycle learning rate schedule.

  Over `total_steps` the rate rises with a cosine from `max / 10` to `max`
  during the first 30% of steps, falls back to `max / 10` during the next
  30%, then decays to `max / 100` over the rest. Returns a function of the
  step count usable as a Polaris `learning_rate`.

  ## Options

    * `:pct_start` - share of steps in each of the first two phases, default `0.3`
    * `:div_factor` - `max / initial`, default `10.0`
    * `:final_div_factor` - `initial / final`, default `10.0`

  """
  @spec one_cycle(float(), pos_integer(), keyword()) :: (Nx.Tensor.t() -> Nx.Tensor.t())
  def one_cycle(max_learning_rate, total_steps, opts \\ []) do
    pct_start = Keyword.get(opts, :pct_start, 0.3)
    div_factor = Keyword.get(opts, :div_factor, 10.0)
    final_div_factor = Keyword.get(opts, :final_div_factor, 10.0)

    initial = max_learning_rate / div_factor
    final = initial / final_div_factor
    phase_1_end = max(pct_start * total_steps, 1.0)
    phase_2_end = max(2 * pct_start * total_steps, phase_1_end + 1.0)
    total = max(total_steps * 1.0, phase_2_end + 1.0)

    fn step ->
      apply_one_cycle(step,
        max: max_learning_rate,
        initial: initial,
        final: final,
        phase_1_end: phase_1_end,
        phase_2_end: phase_2_end,
        total: total
      )
    end
  end

  defnp apply_one_cycle(step, opts) do
    step = Nx.as_type(step, :f32)

    warmup = cosine_anneal(opts[:initial], opts[:max], step / opts[:phase_1_end])

    cooldown =
      cosine_anneal(
        opts[:max],
        opts[:initial],
        (step - opts[:phase_1_end]) / (opts[:phase_2_end] - opts[:phase_1_end])
      )

    tail =
      cosine_anneal(
        opts[:initial],
        opts[:final],
        (step - opts[:phase_2_end]) / (opts[:total] - opts[:phase_2_end])
      )

    Nx.select(
      step < opts[:phase_1_end],
      warmup,
      Nx.select(step < opts[:phase_2_end], cooldown, tail)
    )
  end

  defnp cosine_anneal(start, finish, pct) do
    pct = Nx.clip(pct, 0.0, 1.0)
    finish + (start - finish) / 2 * (1 + Nx.cos(Nx.Constants.pi() * pct))
  end

  defp build_optimizer(config, learning_rate, total_steps) do
    rate =
      case config[:schedule] || :constant do
        :one_cycle -> one_cycle(learning_rate, total_steps)
        :constant -> learning_rate
      end

    case config[:optimizer] || :adam do
      :adamw -> Polaris.Optimizers.adamw(learning_rate: rate, decay: @adamw_weight_decay)
      :adam -> Polaris.Optimizers.adam(learning_rate: rate)
    end
  end

  defp build_options(nil), do: []
  defp build_options(seed), do: [seed: seed]

  @doc """
  Picks a batch size from the number of training rows.

  Follows NeuralProphet's heuristic: `2 ** (2 + floor(log10(n)))`, clamped
  to the range 16..512. So 100 rows gives 16, 1,000 rows gives 32 and
  100,000 rows gives 128.

  ## Examples

      iex> Soothsayer.Trainer.auto_batch_size(1826)
      32

  """
  @spec auto_batch_size(pos_integer()) :: pos_integer()
  def auto_batch_size(n_rows) when n_rows > 0 do
    magnitude = n_rows |> :math.log10() |> floor()

    (2 ** (2 + magnitude))
    |> max(@min_auto_batch_size)
    |> min(@max_auto_batch_size)
  end

  @doc """
  Splits the training data into shuffled minibatches.

  Rows are shuffled, then cut into batches of exactly `batch_size` rows.
  Any leftover rows that don't fill a batch are dropped for this pass, so
  every batch has the same shape and the compiled train step is reused.
  Since rows are reshuffled on every call, the dropped rows differ between
  epochs.

  ## Parameters

    * `x` - A map of input tensors, all with the same number of samples
      on the leading axis.
    * `y` - A tensor of target values with the same number of samples.
    * `batch_size` - Samples per batch. Must not exceed the number of samples.

  ## Returns

    A list of `{x_batch, y_batch}` tuples.

  ## Examples

      iex> x = %{"a" => Nx.iota({100, 2}), "b" => Nx.iota({100, 3})}
      iex> y = Nx.iota({100, 1})
      iex> batches = Soothsayer.Trainer.batches(x, y, 32)
      iex> length(batches)
      3

  """
  @spec batches(%{String.t() => Nx.Tensor.t()}, Nx.Tensor.t(), pos_integer()) ::
          [{%{String.t() => Nx.Tensor.t()}, Nx.Tensor.t()}]
  def batches(x, y, batch_size) do
    n_rows = Nx.axis_size(y, 0)
    order = 0..(n_rows - 1) |> Enum.shuffle() |> Nx.tensor(type: :s64)

    y_batches = y |> Nx.take(order, axis: 0) |> Nx.to_batched(batch_size, leftover: :discard)

    x_batches =
      Enum.map(x, fn {key, tensor} ->
        tensor
        |> Nx.take(order, axis: 0)
        |> Nx.to_batched(batch_size, leftover: :discard)
        |> Stream.map(&{key, &1})
      end)

    [y_batches | x_batches]
    |> Stream.zip()
    |> Enum.map(fn batch ->
      [y_batch | x_batch] = Tuple.to_list(batch)
      {Map.new(x_batch), y_batch}
    end)
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
  @spec compute_l1_penalty(Soothsayer.Model.model_state(), list(String.t())) :: Nx.Tensor.t()
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

  defp train_standard(network, x, y, epochs, batch_size, initial_params, config, optimizer) do
    # Axon.Loop re-enumerates the data on every epoch, and Stream.resource
    # calls its start function each time, so each epoch gets a fresh shuffle.
    data =
      Stream.resource(
        fn -> batches(x, y, batch_size) end,
        fn
          [] -> {:halt, []}
          [batch | rest] -> {[batch], rest}
        end,
        fn _ -> :ok end
      )

    quantiles = config[:quantiles] || []

    network
    |> Axon.Loop.trainer(&loss(&1, &2, quantiles), optimizer)
    |> Axon.Loop.run(data, initial_params, epochs: epochs, compiler: EXLA)
  end

  @doc """
  The training loss: Huber loss on the median forecast plus, for each
  configured quantile, the pinball loss on that quantile's head.

  ## Parameters

    * `targets` - `{samples, forecast_steps}` target values
    * `predictions` - The network output map, with `:combined` and,
      when quantiles are configured, a `:quantiles` tuple
    * `quantiles` - The sorted quantile list from the config
    * `sample_weight` - Optional `{samples, forecast_steps}` weights, see
      `recency_weights/2`. Every element of every term is multiplied by
      its weight before the mean.

  """
  @spec loss(Nx.Tensor.t(), map(), list(float()), Nx.Tensor.t() | nil) :: Nx.Tensor.t()
  def loss(targets, predictions, quantiles, sample_weight \\ nil) do
    base_loss =
      targets
      |> Axon.Losses.huber(predictions.combined, reduction: :none)
      |> weighted_mean(sample_weight)

    quantile_predictions =
      case Map.get(predictions, :quantiles) do
        nil -> []
        tuple -> Tuple.to_list(tuple)
      end

    quantiles
    |> Enum.zip(quantile_predictions)
    |> Enum.reduce(base_loss, fn {quantile, prediction}, total ->
      Nx.add(total, Quantiles.pinball_loss(targets, prediction, quantile, sample_weight))
    end)
  end

  @doc false
  def weighted_mean(elementwise, nil), do: Nx.mean(elementwise)

  def weighted_mean(elementwise, sample_weight),
    do: Nx.mean(Nx.multiply(elementwise, sample_weight))

  defp train_custom(network, x, y, epochs, batch_size, initial_params, config, optimizer) do
    # The weights ride along as a jit argument. Captured in a closure they
    # would sit on a backend the gradient can't reach.
    weights = Map.new(regularization_terms(initial_params, config))
    local_weights = Map.new(local_terms(initial_params, config))

    {_init_fn, predict_fn} = Axon.build(network)
    {init_optim_fn, update_fn} = optimizer

    objective_fn = build_objective_fn(predict_fn, config[:quantiles] || [])
    jit_train_step = EXLA.jit(build_train_step_fn(objective_fn, update_fn))

    train_step = fn params, opt_state, x_batch, y_batch ->
      jit_train_step.(params, opt_state, x_batch, y_batch, {weights, local_weights})
    end

    initial_opt_state = init_optim_fn.(initial_params)

    run_training_loop(epochs, batch_size, initial_params, initial_opt_state, x, y, train_step)
  end

  @doc """
  The L1 terms a config asks for, as `{layer_name, weights}` pairs where
  `weights` holds one lambda per kernel row (input column), so a layer whose
  columns belong to different events or regressors can penalize each with
  its own strength. Layers the params don't have are skipped, and a config
  with no regularization anywhere gives `[]`.

  `ar.regularization` covers every `ar_dense*` layer, `trend.regularization`
  the `trend_dense` layer except its first row, the base slope, so only the
  slope changes and jumps are pulled toward zero, the way NeuralProphet
  penalizes its trend deltas and not `k`. `seasonality.regularization`
  covers every seasonal layer,
  and events and regressors bring their per-column lambdas from
  `Soothsayer.Events.regularization_weights/1` and
  `Soothsayer.Regressors.regularization_weights/1`.
  """
  @spec regularization_terms(Soothsayer.Model.model_state(), map()) ::
          list({String.t(), Nx.Tensor.t()})
  def regularization_terms(params, config) do
    layer_names = Map.keys(params.data)

    prefix_terms =
      for {prefix, lambda} <-
            [
              {"ar_dense", get_in(config, [:ar, :regularization])},
              {"trend_dense", get_in(config, [:trend, :regularization])}
            ] ++ seasonality_lambdas(config),
          lambda != nil and lambda > 0,
          name <- layer_names,
          String.starts_with?(name, prefix) do
        {name, prefix_weights(prefix, params.data[name]["kernel"], lambda)}
      end

    column_terms =
      for {name, lambdas} <-
            Enum.concat(
              Events.regularization_weights(config[:events] || %{}),
              Regressors.regularization_weights(config)
            ),
          name in layer_names,
          Enum.any?(lambdas, &(&1 > 0)) do
        {name, Nx.tensor(lambdas, type: :f32) |> Nx.reshape({:auto, 1})}
      end

    prefix_terms ++ column_terms
  end

  defp seasonality_lambdas(config) do
    lambda = get_in(config, [:seasonality, :regularization])
    for period <- Seasonality.periods(config), do: {"#{period}_dense", lambda}
  end

  # One lambda per kernel row; a local kernel {series, rows, 1} broadcasts
  # the same rows over its series axis.
  defp uniform_weights(kernel, lambda) do
    Nx.broadcast(Nx.tensor(lambda, type: :f32), {Nx.axis_size(kernel, -2), 1})
  end

  # The trend kernel's first row is the base slope, which the penalty leaves alone
  defp prefix_weights("trend_dense", kernel, lambda) do
    Nx.put_slice(uniform_weights(kernel, lambda), [0, 0], Nx.tensor([[0.0]], type: :f32))
  end

  defp prefix_weights(_prefix, kernel, lambda), do: uniform_weights(kernel, lambda)

  @doc """
  The local regularization terms of a config, `{layer_name, lambda}` for
  every layer with one kernel per series when `series.local_regularization`
  is set: the penalty pulls each series' kernel toward the mean kernel
  across series, `lambda * mean((kernel - mean over series)^2)`, so the
  series can differ without wandering off (NeuralProphet's glocal mode).
  The bias is left alone, so under `normalize: :global` levels stay apart.
  """
  @spec local_terms(Soothsayer.Model.model_state(), map()) :: list({String.t(), number()})
  def local_terms(params, config) do
    case get_in(config, [:series, :local_regularization]) do
      lambda when is_number(lambda) and lambda > 0 ->
        for name <- Soothsayer.Series.local_layers(config), Map.has_key?(params.data, name) do
          {name, lambda}
        end

      _off ->
        []
    end
  end

  @doc """
  The local regularization penalty of `params` for `{layer_name => lambda}`.
  """
  @spec local_penalty(Soothsayer.Model.model_state(), %{String.t() => number()}) :: Nx.Tensor.t()
  def local_penalty(_params, lambdas) when map_size(lambdas) == 0 do
    Nx.tensor(0.0)
  end

  def local_penalty(params, lambdas) do
    lambdas
    |> Enum.map(fn {layer_name, lambda} ->
      kernel = params.data[layer_name]["kernel"]
      spread = Nx.subtract(kernel, Nx.mean(kernel, axes: [0], keep_axes: true))
      Nx.multiply(lambda, Nx.mean(Nx.pow(spread, 2)))
    end)
    |> Enum.reduce(&Nx.add/2)
  end

  @doc """
  The weighted L1 penalty for `regularization_terms/2`: for every term,
  `sum(|kernel| * weights)` with the weights broadcast over the kernel's
  output columns.
  """
  @spec weighted_l1_penalty(
          Soothsayer.Model.model_state(),
          list({String.t(), Nx.Tensor.t()}) | %{String.t() => Nx.Tensor.t()}
        ) :: Nx.Tensor.t()
  def weighted_l1_penalty(params, terms) do
    if Enum.empty?(terms) do
      Nx.tensor(0.0)
    else
      terms
      |> Enum.map(fn {layer_name, weights} ->
        kernel = params.data[layer_name]["kernel"]
        Nx.sum(Nx.multiply(Nx.abs(kernel), weights))
      end)
      |> Enum.reduce(&Nx.add/2)
    end
  end

  defp build_objective_fn(predict_fn, quantiles) do
    fn params, x_input, y_target, {weights, local_weights} ->
      predictions = predict_fn.(params, x_input)
      base_loss = loss(y_target, predictions, quantiles, Map.get(x_input, "sample_weight"))

      base_loss
      |> Nx.add(weighted_l1_penalty(params, weights))
      |> Nx.add(local_penalty(params, local_weights))
    end
  end

  defp build_train_step_fn(objective_fn, update_fn) do
    fn params, opt_state, x_input, y_target, weights ->
      {loss, grads} =
        Nx.Defn.value_and_grad(params, fn p -> objective_fn.(p, x_input, y_target, weights) end)

      {updates, new_opt_state} = update_fn.(grads, opt_state, params)
      new_params = Polaris.Updates.apply_updates(params, updates)
      {loss, new_params, new_opt_state}
    end
  end

  defp run_training_loop(
         epochs,
         batch_size,
         initial_params,
         initial_opt_state,
         x,
         y,
         jit_train_step
       ) do
    log_interval = max(div(epochs, 10), 1)

    {final_params, _final_opt_state} =
      Enum.reduce(1..epochs, {initial_params, initial_opt_state}, fn epoch, {params, opt_state} ->
        {final_loss, new_params, new_opt_state} =
          Enum.reduce(batches(x, y, batch_size), {nil, params, opt_state}, fn {x_batch, y_batch},
                                                                              {_loss, p, os} ->
            jit_train_step.(p, os, x_batch, y_batch)
          end)

        if rem(epoch - 1, log_interval) == 0 do
          IO.puts("Epoch: #{epoch - 1}, loss: #{Nx.to_number(final_loss)}")
        end

        {new_params, new_opt_state}
      end)

    final_params
  end
end
