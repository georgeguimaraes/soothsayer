defmodule Soothsayer.Preprocessor do
  @moduledoc """
  Provides data preprocessing functionality for the Soothsayer forecasting model.
  """

  alias Explorer.DataFrame
  alias Explorer.Series

  @doc """
  Prepares the input data by adding Fourier terms for yearly and weekly seasonality based on the provided configuration.

  ## Parameters

    * `df` - An `Explorer.DataFrame` containing the input data.
    * `y_column` - The name of the target variable column.
    * `ds_column` - The name of the date column.
    * `seasonality_config` - A map containing the seasonality configuration.

  ## Returns

    An `Explorer.DataFrame` with additional columns for Fourier terms based on the seasonality configuration.

  ## Examples

      iex> df = Explorer.DataFrame.new(%{"ds" => [...], "y" => [...]})
      iex> seasonality_config = %{yearly: %{enabled: true, fourier_terms: 6}, weekly: %{enabled: true, fourier_terms: 3}}
      iex> prepared_df = Soothsayer.Preprocessor.prepare_data(df, "y", "ds", seasonality_config)
      #Explorer.DataFrame<...>

  """
  @spec prepare_data(Explorer.DataFrame.t(), String.t() | nil, String.t(), map()) :: Explorer.DataFrame.t()
  def prepare_data(df, y_column, ds_column, seasonality_config) do
    df =
      if seasonality_config.yearly.enabled do
        add_fourier_terms(df, ds_column, :yearly, seasonality_config.yearly.fourier_terms)
      else
        df
      end

    df =
      if seasonality_config.weekly.enabled do
        add_fourier_terms(df, ds_column, :weekly, seasonality_config.weekly.fourier_terms)
      else
        df
      end

    if y_column do
      DataFrame.select(df, [y_column | df.names -- [y_column]])
    else
      df
    end
  end

  defp add_fourier_terms(df, ds_column, period_type, fourier_terms) do
    date_series = df[ds_column]

    t =
      case period_type do
        :yearly ->
          days_in_year =
            date_series
            |> Series.to_list()
            |> Enum.map(fn date ->
              if Date.leap_year?(date), do: 366.0, else: 365.0
            end)
            |> Series.from_list()

          Series.day_of_year(date_series)
          |> Series.cast(:float64)
          |> Series.divide(days_in_year)

        :weekly ->
          Series.day_of_week(date_series)
          |> Series.cast(:float64)
          |> Series.divide(Series.from_list(List.duplicate(7.0, Series.size(date_series))))
      end

    result_df = Enum.reduce(1..fourier_terms, df, fn i, acc_df ->
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

    result_df
  end
end
