defmodule Soothsayer.Regressors do
  @moduledoc """
  Future regressor component for Soothsayer models.

  A future regressor is an external variable whose value is known for every
  date you want to forecast: temperature from a weather forecast, planned
  marketing spend, a price schedule. Each regressor becomes one input column
  feeding a linear layer, so its learned coefficient reads directly as the
  effect of one normalized unit of the regressor on the forecast.

  Regressors are configured as a list of column names. The training
  dataframe must contain those columns, and so must the dataframe passed as
  `regressors:` to `Soothsayer.predict/3`.
  """

  alias Explorer.DataFrame
  alias Explorer.Series

  @layer_name "regressors_dense"

  # Network Building

  @doc """
  Creates the Axon input node for the regressors component.

  ## Returns

    An Axon input node with one column per regressor, `nil` when none are configured.

  """
  @spec build_network_input(map()) :: Axon.t() | nil
  def build_network_input(%{regressors: [_ | _] = names}) do
    Axon.input("regressors", shape: {nil, length(names)})
  end

  def build_network_input(_config), do: nil

  @doc """
  Builds the regressors component layer.

  ## Returns

    An Axon dense layer when regressors are configured, `Axon.constant(0)` otherwise.

  """
  @spec build_component(Axon.t() | nil, map()) :: Axon.t()
  def build_component(nil, _config), do: Axon.constant(0)

  def build_component(input, %{regressors: [_ | _]}) do
    Axon.dense(input, 1, activation: :linear, name: @layer_name)
  end

  def build_component(_input, _config), do: Axon.constant(0)

  # Feature Engineering

  @doc """
  Builds the regressors input tensor for a list of dates.

  Looks each date up in `dataframe` (which needs a "ds" column plus one
  column per regressor) and stacks the regressor values in config order.

  Raises `ArgumentError` when a regressor column is missing or when any date
  has no row, since a forecast that silently fills in zeros for an unknown
  regressor value would be wrong without saying so.

  ## Examples

      iex> dataframe = Explorer.DataFrame.new(%{"ds" => [~D[2023-01-01], ~D[2023-01-02]], "temperature" => [20.0, 22.5]})
      iex> Soothsayer.Regressors.build_features([~D[2023-01-02]], dataframe, ["temperature"])
      #Nx.Tensor<
        f32[1][1]
        [
          [22.5]
        ]
      >

  """
  @spec build_features(list(Date.t()), DataFrame.t(), list(String.t())) :: Nx.Tensor.t()
  def build_features(dates, %DataFrame{} = dataframe, names) do
    validate_columns!(dataframe, names)

    values_by_date =
      dataframe["ds"]
      |> Series.to_list()
      |> Enum.zip(rows(dataframe, names))
      |> Map.new()

    rows =
      Enum.map(dates, fn date ->
        case Map.fetch(values_by_date, date) do
          {:ok, values} ->
            values

          :error ->
            raise ArgumentError,
                  "Regressor values for #{Date.to_iso8601(date)} are missing. " <>
                    "The regressors dataframe must cover every date being predicted, " <>
                    "including the days between the last observation and the forecast."
        end
      end)

    rows |> Nx.tensor() |> Nx.as_type({:f, 32})
  end

  @doc """
  Raises `ArgumentError` unless `dataframe` has every regressor column.
  """
  @spec validate_columns!(DataFrame.t(), list(String.t())) :: :ok
  def validate_columns!(%DataFrame{} = dataframe, names) do
    columns = DataFrame.names(dataframe)

    for name <- names, name not in columns do
      raise ArgumentError,
            "Regressor column #{inspect(name)} not found. Available columns: #{inspect(columns)}"
    end

    :ok
  end

  defp rows(dataframe, names) do
    names
    |> Enum.map(fn name -> dataframe[name] |> Series.cast({:f, 64}) |> Series.to_list() end)
    |> Enum.zip_with(& &1)
  end

  # Weight Extraction

  @doc """
  Extracts the learned regressor coefficients from a fitted model.

  Coefficients are in normalized units: the change in normalized y for a one
  standard deviation change in the regressor.

  ## Returns

    A map from regressor name to coefficient.

  """
  @spec get_effects(Soothsayer.Model.t()) :: %{String.t() => float()}
  def get_effects(%Soothsayer.Model{} = model) do
    names = model.config[:regressors] || []

    if names == [] do
      raise ArgumentError, "No regressors configured on this model"
    end

    unless model.params do
      raise ArgumentError, "Model has not been fitted yet"
    end

    coefficients = model.params.data[@layer_name]["kernel"] |> Nx.to_flat_list()

    Enum.zip(names, coefficients) |> Map.new()
  end
end
