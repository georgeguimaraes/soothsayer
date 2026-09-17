defmodule Soothsayer.Conformal do
  @moduledoc """
  Split conformal prediction: intervals calibrated on held out data.

  The quantile heads give intervals shaped by the training noise, with no
  guarantee on how many future points they cover. Calibration fixes the
  width after the fact. The fitted model forecasts a calibration frame it
  has not seen, every row gets a nonconformity score saying how wrong the
  forecast was, and the `1 - alpha` quantile of those scores becomes the
  width correction `q_hat`. With exchangeable data the resulting interval
  covers a future point with probability at least `1 - alpha`.

  Two methods, as in NeuralProphet:

    * `:naive` scores `|y - yhat|` and builds `yhat -+ q_hat` around the
      point forecast. No quantiles needed.
    * `:cqr` (conformalized quantile regression) scores how far `y` falls
      outside the interval between the lowest and highest configured
      quantile, `max(lower - y, y - upper)`, and pushes that interval out
      by `q_hat`. With `alpha: {alpha_lower, alpha_upper}` the two sides
      get their own scores and corrections.

  Scores are kept per forecast step, so a multi-step model gets one
  `q_hat` per step ahead. Rows forecast further out than the model's
  `forecast_steps` were never calibrated and use the last step's `q_hat`.

  `q_hat` is the `ceil((n + 1) * (1 - alpha)) / n` empirical quantile of
  the sorted scores, the finite-sample version of split conformal.
  NeuralProphet takes `scores[-int(n * alpha)]`, which is slightly narrower
  and falls back to the smallest score when `n * alpha < 1`.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Backtest
  alias Soothsayer.Quantiles

  @type calibration :: %{
          alpha: float() | {float(), float()},
          method: :naive | :cqr,
          q_hat: %{pos_integer() => float() | {float(), float()}},
          count: pos_integer()
        }

  @doc """
  Calibrates a fitted model on a frame with `ds` and `y` covering the
  period right after the training data. Returns the calibration map that
  `Soothsayer.calibrate/3` stores on the model.

  Options: `:alpha` (default `0.1`), `:method` (default `:naive`), and
  `:events` and `:regressors` frames for the calibration dates, as for
  `Soothsayer.predict/3`.
  """
  @spec calibrate(Soothsayer.Model.t(), DataFrame.t(), keyword()) :: calibration()
  def calibrate(%Soothsayer.Model{} = model, %DataFrame{} = frame, opts \\ []) do
    alpha = Keyword.get(opts, :alpha, 0.1)
    method = Keyword.get(opts, :method, :naive)
    validate!(model, alpha, method)

    predictions =
      Backtest.rolling_predictions(model, frame,
        events: Keyword.get(opts, :events),
        regressors: Keyword.get(opts, :regressors, frame)
      )

    quantiles = model.config.quantiles

    q_hat =
      predictions["step"]
      |> Series.distinct()
      |> Series.to_list()
      |> Map.new(fn step ->
        rows = DataFrame.filter_with(predictions, &Series.equal(&1["step"], step))
        {step, step_q_hat(rows, method, alpha, quantiles)}
      end)

    %{alpha: alpha, method: method, q_hat: q_hat, count: DataFrame.n_rows(predictions)}
  end

  defp step_q_hat(rows, :naive, alpha, _quantiles) do
    rows["y"]
    |> Series.subtract(rows["yhat"])
    |> Series.abs()
    |> Series.to_list()
    |> q_hat(alpha)
  end

  defp step_q_hat(rows, :cqr, alpha, quantiles) do
    {lower, upper} = quantile_bounds(rows, quantiles)
    below = Series.subtract(lower, rows["y"]) |> Series.to_list()
    above = Series.subtract(rows["y"], upper) |> Series.to_list()

    case alpha do
      {alpha_lower, alpha_upper} ->
        {q_hat(below, alpha_lower), q_hat(above, alpha_upper)}

      alpha ->
        below |> Enum.zip_with(above, &max/2) |> q_hat(alpha)
    end
  end

  defp quantile_bounds(rows, quantiles) do
    {rows[Quantiles.column_name(Enum.min(quantiles))],
     rows[Quantiles.column_name(Enum.max(quantiles))]}
  end

  @doc """
  The width correction for a list of nonconformity scores: the
  `ceil((n + 1) * (1 - alpha))`-th smallest score. Raises when there are
  too few scores for that rank to exist, which needs at least
  `ceil(1 / alpha - 1)` rows.

  ## Examples

      iex> Soothsayer.Conformal.q_hat(Enum.to_list(1..20), 0.1)
      19

  """
  @spec q_hat(list(number()), float()) :: number()
  def q_hat(scores, alpha) do
    n = length(scores)
    rank = ceil((n + 1) * (1 - alpha))

    if rank > n do
      raise ArgumentError,
            "Calibration needs at least #{ceil(1 / alpha - 1)} usable rows for alpha #{alpha}, " <>
              "got #{n}"
    end

    scores |> Enum.sort() |> Enum.at(rank - 1)
  end

  @doc """
  Lower and upper interval bounds for predictions, from the calibration
  stored on the model. `steps` is the `{n, 1}` step-ahead tensor of the
  rows, `combined` the point forecast and `quantiles` the denormalized
  quantile map, all as returned by `Soothsayer.predict_components/3`.
  """
  @spec bounds(calibration(), Nx.Tensor.t(), Nx.Tensor.t(), %{float() => Nx.Tensor.t()}) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def bounds(%{method: :naive} = calibration, steps, combined, _quantiles) do
    q = per_row(calibration, steps, & &1)
    {Nx.subtract(combined, q), Nx.add(combined, q)}
  end

  def bounds(%{method: :cqr} = calibration, steps, _combined, quantiles) do
    {lower_quantile, upper_quantile} = Enum.min_max(Map.keys(quantiles))
    q_lower = per_row(calibration, steps, &elem(as_pair(&1), 0))
    q_upper = per_row(calibration, steps, &elem(as_pair(&1), 1))
    {Nx.subtract(quantiles[lower_quantile], q_lower), Nx.add(quantiles[upper_quantile], q_upper)}
  end

  defp as_pair({lower, upper}), do: {lower, upper}
  defp as_pair(q), do: {q, q}

  # One q_hat per row from its step, rows past the calibrated steps take
  # the last one.
  defp per_row(%{q_hat: q_hat}, steps, pick) do
    last_step = q_hat |> Map.keys() |> Enum.max()
    by_step = Enum.map(1..last_step, &pick.(q_hat[&1]))
    index = steps |> Nx.subtract(1) |> Nx.clip(0, last_step - 1)
    Nx.take(Nx.tensor(by_step, type: {:f, 32}), index)
  end

  defp validate!(model, alpha, method) do
    unless model.params do
      raise ArgumentError, "Model has not been fitted yet"
    end

    unless method in [:naive, :cqr] do
      raise ArgumentError, "method must be :naive or :cqr, got #{inspect(method)}"
    end

    if method == :cqr and model.config.quantiles == [] do
      raise ArgumentError, "method: :cqr needs quantiles configured on the model"
    end

    validate_alpha!(alpha, method)
  end

  defp validate_alpha!(alpha, _method) when is_number(alpha) and alpha > 0 and alpha < 1, do: :ok

  defp validate_alpha!({lower, upper} = pair, method)
       when is_number(lower) and lower > 0 and lower < 1 and is_number(upper) and upper > 0 and
              upper < 1 do
    if method == :naive do
      raise ArgumentError,
            "alpha #{inspect(pair)} as a {lower, upper} pair needs method: :cqr, " <>
              ":naive intervals are symmetric"
    end

    :ok
  end

  defp validate_alpha!(other, _method) do
    raise ArgumentError,
          "alpha must be a number in (0, 1) or a {lower, upper} pair of them, got #{inspect(other)}"
  end
end
