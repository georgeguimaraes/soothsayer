defmodule Soothsayer.Seasonality do
  @moduledoc """
  Seasonality component for Soothsayer forecasting models.

  Handles network building and feature engineering for yearly and weekly seasonality
  using Fourier series decomposition.
  """

  alias Explorer.DataFrame
  alias Explorer.Series

  # Calendar constants
  @days_per_regular_year 365.0
  @days_per_leap_year 366.0
  @days_per_week 7.0

  # Network Building

  @doc """
  Creates Axon input nodes for seasonality components.

  ## Parameters

    * `config` - Model configuration map with `:seasonality` key.

  ## Returns

    A map with `:yearly` and `:weekly` keys containing Axon input nodes.

  """
  @spec build_inputs(map()) :: %{yearly: Axon.t(), weekly: Axon.t()}
  def build_inputs(config) do
    yearly_terms = get_in(config, [:seasonality, :yearly, :fourier_terms]) || 0
    weekly_terms = get_in(config, [:seasonality, :weekly, :fourier_terms]) || 0

    %{
      yearly: Axon.input("yearly", shape: {nil, yearly_terms * 2}),
      weekly: Axon.input("weekly", shape: {nil, weekly_terms * 2})
    }
  end

  @doc """
  Builds seasonality component layers.

  ## Parameters

    * `inputs` - Map of Axon input nodes from `build_inputs/1`.
    * `config` - Model configuration map.

  ## Returns

    A map with `:yearly` and `:weekly` keys containing Axon layers.

  """
  @spec build_components(%{yearly: Axon.t(), weekly: Axon.t()}, map()) ::
          %{yearly: Axon.t(), weekly: Axon.t()}
  def build_components(inputs, config) do
    %{
      yearly: build_period_component(inputs.yearly, config, :yearly),
      weekly: build_period_component(inputs.weekly, config, :weekly)
    }
  end

  defp build_period_component(input, config, period) do
    enabled = get_in(config, [:seasonality, period, :enabled])

    if enabled do
      Axon.dense(input, 1, activation: :linear, name: "#{period}_dense")
    else
      Axon.constant(0)
    end
  end

  # Feature Engineering

  @doc """
  Adds Fourier feature columns to a DataFrame for seasonality modeling.

  ## Parameters

    * `df` - An `Explorer.DataFrame` containing the input data.
    * `ds_column` - The name of the date column.
    * `seasonality_config` - A map containing the seasonality configuration.

  ## Returns

    An `Explorer.DataFrame` with additional columns for Fourier terms.

  ## Examples

      iex> df = Explorer.DataFrame.new(%{"ds" => [~D[2023-01-01]], "y" => [1.0]})
      iex> config = %{yearly: %{enabled: true, fourier_terms: 2}, weekly: %{enabled: false, fourier_terms: 2}}
      iex> result = Soothsayer.Seasonality.add_fourier_features(df, "ds", config)
      iex> "yearly_sin_1" in result.names
      true

  """
  @spec add_fourier_features(Explorer.DataFrame.t(), String.t(), map()) :: Explorer.DataFrame.t()
  def add_fourier_features(df, ds_column, seasonality_config) do
    df =
      if seasonality_config.yearly.enabled do
        add_fourier_terms(df, ds_column, :yearly, seasonality_config.yearly.fourier_terms)
      else
        df
      end

    if seasonality_config.weekly.enabled do
      add_fourier_terms(df, ds_column, :weekly, seasonality_config.weekly.fourier_terms)
    else
      df
    end
  end

  defp add_fourier_terms(df, ds_column, period_type, fourier_terms) do
    date_series = df[ds_column]

    t =
      case period_type do
        :yearly ->
          days_in_year = compute_days_in_year(date_series)

          Series.day_of_year(date_series)
          |> Series.cast({:f, 64})
          |> Series.divide(days_in_year)

        :weekly ->
          Series.day_of_week(date_series)
          |> Series.cast({:f, 64})
          |> Series.divide(
            Series.from_list(List.duplicate(@days_per_week, Series.size(date_series)))
          )
      end

    Enum.reduce(1..fourier_terms, df, fn i, acc_df ->
      acc_df
      |> DataFrame.put(
        "#{period_type}_sin_#{i}",
        Series.sin(t |> Series.multiply(2 * :math.pi() * i))
      )
      |> DataFrame.put(
        "#{period_type}_cos_#{i}",
        Series.cos(t |> Series.multiply(2 * :math.pi() * i))
      )
    end)
  end

  defp compute_days_in_year(date_series) do
    date_series
    |> Series.to_list()
    |> Enum.map(&days_for_date/1)
    |> Series.from_list()
  end

  defp days_for_date(date) do
    if Date.leap_year?(date), do: @days_per_leap_year, else: @days_per_regular_year
  end
end
