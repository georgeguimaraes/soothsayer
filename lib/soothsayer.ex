defmodule Soothsayer do
  @moduledoc """
  Soothsayer is an Elixir port of NeuralProphet, providing time series forecasting capabilities.
  """

  alias Explorer.DataFrame
  alias Explorer.Series

  @doc """
  Creates a new Soothsayer model with the given options.
  """
  def new(opts \\ []) do
    params =
      %{
        # Add default parameters here
      }

    %Soothsayer.Model{params: params, state: %{}}
  end

  def fit(%Soothsayer.Model{} = model, %DataFrame{} = data, freq) when is_binary(freq) do
    # Extract Series from DataFrame
    date_series = DataFrame.pull(data, "date")
    target_series = DataFrame.pull(data, "y")

    # Convert Series to tensors
    date_tensor = Series.to_tensor(date_series)
    target_tensor = Series.to_tensor(target_series)

    # Cast target_tensor to float32
    target_tensor = Nx.as_type(target_tensor, {:f, 32})

    # Placeholder for model training
    # In reality, this would involve complex calculations using Nx and Axon
    trained_state = %{
      date_tensor: date_tensor,
      target_tensor: target_tensor,
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

  def predict(%Soothsayer.Model{state: state} = model, %DataFrame{} = data) do
    # Extract Series from DataFrame
    date_series = DataFrame.pull(data, "date")

    # Get the last value from the target tensor
    last_value = state.target_tensor |> Nx.to_flat_list() |> List.last()

    # Create a new tensor with the same shape as the target, filled with the last value
    predictions = Nx.broadcast(Nx.tensor(last_value), Nx.shape(state.target_tensor))

    # Convert predictions back to a DataFrame
    forecast =
      DataFrame.new(%{
        "date" => date_series,
        "yhat" => Series.from_tensor(predictions)
      })

    forecast
  end
end
