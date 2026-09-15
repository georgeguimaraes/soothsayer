defmodule Soothsayer.Backtest do
  @moduledoc """
  Rolling-origin evaluation of a model configuration on held out data.

  Splits the data into a training and a validation part, fits the model on
  the training part, then walks through the validation part one origin at a
  time: from each origin it forecasts the next `horizon` dates using only
  what was observed up to that origin, and compares against the actual
  values. Every validation date is therefore forecast once from each of the
  `horizon` steps before it, which is the same protocol NeuralProphet uses
  for its validation metrics.

  For models without auto-regression the forecast for a date doesn't depend
  on the origin, so the per-step metrics come out identical.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR

  @type result :: %{
          model: Soothsayer.Model.t(),
          metrics: %{mean_absolute_error: float(), root_mean_squared_error: float()},
          by_step: %{
            pos_integer() => %{mean_absolute_error: float(), root_mean_squared_error: float()}
          },
          predictions: DataFrame.t()
        }

  @doc """
  Runs the backtest.

  ## Options

    * `:validation_fraction` - Share of rows held out at the end, default `0.1`.
      Like NeuralProphet's `split_df`, the validation part is the last
      `max(1, trunc(rows * fraction))` rows.
    * `:horizon` - Steps ahead to forecast from each origin. Defaults to the
      model's `ar.forecast_steps`, or 1 without auto-regression.
    * `:events` - Events dataframe, passed to both fit and predict.
    * `:regressors` - Regressors dataframe for prediction. Defaults to the
      validation part itself, which holds the regressor columns for those dates.

  ## Returns

    A map with the fitted `:model`, overall `:metrics`, `:by_step` metrics
    keyed by step ahead, and a `:predictions` dataframe with columns
    `origin`, `ds`, `step`, `y` and `yhat`. Validation dates whose `y` is
    missing (nil or NaN) can't be scored and are left out of both; they
    still flow into the history each forecast is made from, where they are
    imputed like training data (see `Soothsayer.MissingData`). Forecasts
    that come out NaN, because their lags reach into a gap that couldn't be
    imputed, are left out the same way.

  """
  @spec run(Soothsayer.Model.t(), DataFrame.t(), keyword()) :: result()
  def run(%Soothsayer.Model{} = model, %DataFrame{} = data, opts \\ []) do
    validation_fraction = Keyword.get(opts, :validation_fraction, 0.1)
    horizon = Keyword.get(opts, :horizon, AR.forecast_steps(model.config))
    events = Keyword.get(opts, :events)

    {train, validation} = split(data, validation_fraction)
    regressors = Keyword.get(opts, :regressors, validation)

    fitted_model = Soothsayer.fit(model, train, fit_options(events))

    predictions =
      rolling_predictions(fitted_model, train, validation, horizon, events, regressors)

    %{
      model: fitted_model,
      metrics: metrics(predictions["y"], predictions["yhat"]),
      by_step: metrics_by_step(predictions),
      predictions: predictions
    }
  end

  @doc """
  Splits a dataframe into training and validation parts, validation being
  the last `max(1, trunc(rows * fraction))` rows.
  """
  @spec split(DataFrame.t(), float()) :: {DataFrame.t(), DataFrame.t()}
  def split(%DataFrame{} = data, validation_fraction) do
    rows = DataFrame.n_rows(data)
    validation_rows = max(1, trunc(rows * validation_fraction))
    {DataFrame.head(data, rows - validation_rows), DataFrame.tail(data, validation_rows)}
  end

  @doc """
  Mean absolute error and root mean squared error between two series.
  """
  @spec metrics(Series.t(), Series.t()) :: %{
          mean_absolute_error: float(),
          root_mean_squared_error: float()
        }
  def metrics(actual, predicted) do
    actual = actual |> Series.to_tensor() |> Nx.as_type({:f, 32})
    predicted = predicted |> Series.to_tensor() |> Nx.as_type({:f, 32})

    %{
      mean_absolute_error: Axon.Metrics.mean_absolute_error(actual, predicted) |> Nx.to_number(),
      root_mean_squared_error:
        Axon.Losses.mean_squared_error(actual, predicted, reduction: :mean)
        |> Nx.sqrt()
        |> Nx.to_number()
    }
  end

  defp missing?(value), do: is_nil(value) or value == :nan

  defp fit_options(nil), do: []
  defp fit_options(events), do: [events: events]

  defp rolling_predictions(fitted_model, train, validation, horizon, events, regressors) do
    validation_dates = Series.to_list(validation["ds"])
    actual_by_date = Enum.zip(validation_dates, Series.to_list(validation["y"])) |> Map.new()
    last_training_date = train["ds"] |> Series.to_list() |> List.last()

    rows =
      Enum.flat_map(0..(length(validation_dates) - 1), fn origin_rows ->
        target_dates = Enum.slice(validation_dates, origin_rows, horizon)

        origin_date =
          if origin_rows == 0,
            do: last_training_date,
            else: Enum.at(validation_dates, origin_rows - 1)

        predict_options =
          [regressors: regressors]
          |> maybe_put(:events, events)
          |> maybe_put(:history, if(origin_rows > 0, do: DataFrame.head(validation, origin_rows)))

        predicted =
          fitted_model
          |> Soothsayer.predict_components(Series.from_list(target_dates), predict_options)
          |> Map.fetch!(:combined)
          |> Nx.to_flat_list()

        target_dates
        |> Enum.zip(predicted)
        |> Enum.with_index(1)
        |> Enum.reject(fn {{date, yhat}, _step} ->
          missing?(actual_by_date[date]) or missing?(yhat)
        end)
        |> Enum.map(fn {{date, yhat}, step} ->
          %{origin: origin_date, ds: date, step: step, y: actual_by_date[date], yhat: yhat}
        end)
      end)

    DataFrame.new(%{
      "origin" => Enum.map(rows, & &1.origin),
      "ds" => Enum.map(rows, & &1.ds),
      "step" => Enum.map(rows, & &1.step),
      "y" => Enum.map(rows, & &1.y),
      "yhat" => Enum.map(rows, & &1.yhat)
    })
  end

  defp metrics_by_step(predictions) do
    predictions["step"]
    |> Series.distinct()
    |> Series.to_list()
    |> Map.new(fn step ->
      rows = DataFrame.filter_with(predictions, &Series.equal(&1["step"], step))
      {step, metrics(rows["y"], rows["yhat"])}
    end)
  end

  defp maybe_put(options, _key, nil), do: options
  defp maybe_put(options, key, value), do: Keyword.put(options, key, value)
end
