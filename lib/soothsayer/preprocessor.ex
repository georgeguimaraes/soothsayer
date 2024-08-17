defmodule Soothsayer.Preprocessor do
  alias Explorer.DataFrame
  alias Explorer.Series

  def prepare_data(df, target_column, date_column) do
    df
    |> add_time_features(date_column)
  end

  defp add_time_features(df, date_column) do
    date_series = df[date_column]

    df
    |> DataFrame.put("day_of_year", Series.day_of_year(date_series))
    |> DataFrame.put("month", Series.month(date_series))
    |> DataFrame.put("day_of_month", day_of_month(date_series))
    |> DataFrame.put("day_of_week", Series.day_of_week(date_series))
  end

  defp day_of_month(date_series) do
    Series.transform(date_series, fn date -> date.day end)
  end
end
