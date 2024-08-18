defmodule Soothsayer.Preprocessor do
  alias Explorer.DataFrame
  alias Explorer.Series

  def prepare_data(df, target_column, date_column, seasonality_config) do
    df
    |> add_time_features(date_column)
    |> add_fourier_terms(date_column, seasonality_config)
  end

  defp add_time_features(df, date_column) do
    date_series = df[date_column]

    df
    |> DataFrame.put("year", Series.transform(date_series, & &1.year))
    |> DataFrame.put("month", Series.transform(date_series, & &1.month))
    |> DataFrame.put("day", Series.transform(date_series, & &1.day))
    |> DataFrame.put("day_of_year", Series.day_of_year(date_series))
    |> DataFrame.put("day_of_week", Series.day_of_week(date_series))
  end

  defp add_fourier_terms(df, date_column, seasonality_config) do
    date_series = df[date_column]

    df =
      if seasonality_config.yearly.enabled do
        add_fourier_terms_for_period(
          df,
          date_series,
          :yearly,
          seasonality_config.yearly.fourier_terms
        )
      else
        df
      end

    if seasonality_config.weekly.enabled do
      add_fourier_terms_for_period(
        df,
        date_series,
        :weekly,
        seasonality_config.weekly.fourier_terms
      )
    else
      df
    end
  end

  defp add_fourier_terms_for_period(df, date_series, period_type, fourier_terms) do
    t =
      case period_type do
        :yearly ->
          days_in_year =
            Series.transform(date_series, fn date ->
              if Date.leap_year?(date), do: 366, else: 365
            end)

          Series.day_of_year(date_series)
          |> Series.cast(:float)
          |> Series.divide(days_in_year)

        :weekly ->
          Series.day_of_week(date_series)
          |> Series.cast(:float)
          |> Series.divide(7)
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
end
