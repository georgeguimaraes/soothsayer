defmodule Soothsayer.LaggedRegressors do
  @moduledoc """
  Lagged regressor component for Soothsayer models.

  A lagged regressor is an external variable whose past values help explain
  the target, the way auto-regression uses the target's own past. Yesterday's
  temperature for today's energy price, last week's ad spend for this week's
  sales. Unlike future regressors, lagged regressors are only ever read at or
  before the forecast origin, so they don't need to be known for the dates
  being forecast.

  Configured as a map from column name to `%{lags: n}`:

      lagged_regressors: %{"temperature" => %{lags: 3}}

  Requires auto-regression to be enabled, since the lag windows are built
  from the same origins. The training dataframe must contain the columns,
  and predicting past the training data needs their values up to each block
  origin, passed through the `regressors:` option of `Soothsayer.predict/3`.
  """

  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.AR
  alias Soothsayer.Frequency
  alias Soothsayer.Timestamp

  @layer_name "lagged_regressors_dense"

  # Network Building

  @doc """
  Creates the Axon input node, one column per regressor lag, or `nil` when
  no lagged regressors are configured.
  """
  @spec build_network_input(map()) :: Axon.t() | nil
  def build_network_input(config) do
    case total_lags(config) do
      0 -> nil
      width -> Axon.input("lagged_regressors", shape: {nil, width})
    end
  end

  @doc """
  Builds the lagged regressors layer, `Axon.constant(0)` when there are none.
  """
  @spec build_component(Axon.t() | nil, map()) :: Axon.t()
  def build_component(nil, _config), do: Axon.constant(0)

  def build_component(input, _config) do
    Axon.dense(input, 1, activation: :linear, name: @layer_name)
  end

  @doc """
  The configured lagged regressors as a sorted list of `{name, lags}`.
  """
  @spec specs(map()) :: list({String.t(), pos_integer()})
  def specs(config) do
    config
    |> Map.get(:lagged_regressors, %{})
    |> Enum.map(fn {name, %{lags: lags}} -> {name, lags} end)
    |> Enum.sort()
  end

  @doc """
  Names of the configured lagged regressor columns.
  """
  @spec names(map()) :: list(String.t())
  def names(config), do: config |> specs() |> Enum.map(&elem(&1, 0))

  @doc """
  The largest configured lag, 0 when there are no lagged regressors.
  """
  @spec max_lags(map()) :: non_neg_integer()
  def max_lags(config) do
    config |> specs() |> Enum.map(&elem(&1, 1)) |> Enum.max(fn -> 0 end)
  end

  defp total_lags(config), do: config |> specs() |> Enum.map(&elem(&1, 1)) |> Enum.sum()

  # Feature Engineering

  @doc """
  Builds the training input from the training dataframe for the given
  origins, one lag window per regressor side by side, repeated per forecast
  step in the same row order as `AR.training_rows/4`.
  """
  @spec build_training_rows(DataFrame.t(), map(), list(non_neg_integer()), pos_integer()) ::
          Nx.Tensor.t()
  def build_training_rows(%DataFrame{} = data, config, origin_indices, forecast_steps) do
    config
    |> specs()
    |> Enum.map(fn {name, lags} ->
      values = data[name] |> Series.cast({:f, 32}) |> Series.to_tensor()
      AR.lagged_rows(values, origin_indices, lags, forecast_steps)
    end)
    |> Nx.concatenate(axis: 1)
  end

  @doc """
  Collects the raw regressor values by timestamp for prediction: the
  training values stored on the model, overridden and extended by the
  `regressors:` dataframe when given.
  """
  @spec known_values(map(), DataFrame.t() | nil, map()) ::
          %{String.t() => %{Timestamp.t() => float()}}
  def known_values(training_data, regressors_df, config) do
    training_values = Map.get(training_data, :lagged_regressors, %{})

    Map.new(names(config), fn name ->
      from_frame =
        if regressors_df != nil and name in DataFrame.names(regressors_df) do
          values_by_date(regressors_df, name)
        else
          %{}
        end

      {name, Map.merge(Map.get(training_values, name, %{}), from_frame)}
    end)
  end

  @doc """
  Raw regressor values by timestamp from a dataframe, for storing at fit time.
  """
  @spec values_by_date(DataFrame.t(), String.t()) :: %{Timestamp.t() => float()}
  def values_by_date(%DataFrame{} = dataframe, name) do
    timestamps = Timestamp.from_series(dataframe["ds"])
    values = dataframe[name] |> Series.cast({:f, 64}) |> Series.to_list()
    Enum.zip(timestamps, values) |> Map.new()
  end

  @doc """
  Builds the prediction input for a list of origin timestamps: for each
  regressor, the `lags` values ending at the origin, oldest first, one step
  of `frequency` apart. Raises when a needed timestamp has no value, since a
  lag silently filled with zero would be wrong without saying so.
  """
  @spec build_input(
          %{String.t() => %{Timestamp.input() => float()}},
          list(Timestamp.input()),
          map(),
          Frequency.t()
        ) :: Nx.Tensor.t()
  def build_input(known_values, origins, config, frequency) do
    config
    |> specs()
    |> Enum.map(fn {name, lags} ->
      values = Map.fetch!(known_values, name)

      origins
      |> Enum.map(&window(values, name, &1, lags, frequency))
      |> Nx.tensor()
      |> Nx.as_type({:f, 32})
    end)
    |> Nx.concatenate(axis: 1)
  end

  defp window(values, name, origin, lags, frequency) do
    Enum.map((lags - 1)..0//-1, fn offset ->
      fetch_value!(values, name, Frequency.shift(origin, -offset, frequency))
    end)
  end

  defp fetch_value!(values, name, timestamp) do
    case Map.fetch(values, timestamp) do
      {:ok, value} ->
        value

      :error ->
        raise ArgumentError,
              "Lagged regressor #{inspect(name)} has no value for " <>
                "#{Timestamp.format(timestamp)}. Pass regressors: a dataframe covering " <>
                "the timestamps up to each forecast origin."
    end
  end
end
