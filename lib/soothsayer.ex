defmodule Soothsayer do
  alias Explorer.DataFrame
  alias Explorer.Series

  def new do
    %Soothsayer.Model{}
  end

  def fit(model, data, freq) when is_binary(freq) do
    prepared_data = Soothsayer.Preprocessor.prepare_data(data, "y", "date")

    # For now, we'll just store the prepared data in the model
    trained_state = %{
      prepared_data: prepared_data,
      freq: freq
    }

    metrics = %{
      # placeholder metric
      mse: 0.5,
      # placeholder metric
      mae: 0.3
    }

    updated_model = %{model | state: trained_state}

    {updated_model, metrics}
  end

  def predict(%Soothsayer.Model{state: state}, future_df) do
    # Extract date column
    dates = DataFrame.pull(future_df, "date")

    # Generate placeholder predictions
    num_predictions = DataFrame.n_rows(future_df)
    last_value = state.prepared_data["y"] |> Series.last()
    yhat = Series.from_list(List.duplicate(last_value, num_predictions))

    # Create forecast DataFrame
    DataFrame.new(%{
      "date" => dates,
      "yhat" => yhat
    })
  end
end
