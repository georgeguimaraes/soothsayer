defmodule Soothsayer.Preprocessor do
  alias Explorer.DataFrame
  alias Explorer.Series

  def prepare_data(data, y_column, ds_column, seasonality_config) do
    df = data

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

    DataFrame.select(df, [y_column | df.names -- [y_column]])
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
          |> Series.cast(:float)
          |> Series.divide(days_in_year)

        :weekly ->
          Series.day_of_week(date_series)
          |> Series.cast(:float)
          |> Series.divide(Series.from_list(List.duplicate(7.0, Series.size(date_series))))
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
