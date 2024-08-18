defmodule Soothsayer do
  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Model

  def new do
    Model.new()
  end

  def fit(%Model{} = model, %{} = data) do
    x_series = data["x"]
    y_series = data["y"]

    # Create a stream that generates fresh tensors for each iteration
    train_data =
      Stream.repeatedly(fn ->
        x = x_series |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(0)
        y = y_series |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(0)
        {x, y}
      end)

    trained_model =
      model.nn_model
      |> Axon.Loop.trainer(:mean_squared_error, :adam)
      |> Axon.Loop.run(train_data, %{}, iterations: 100)

    IO.inspect(trained_model)

    %{model | nn_model: trained_model}
  end

  def predict(%Model{nn_model: model}, x) do
    x_tensor = x |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(0)
    Axon.predict(model, %{}, x_tensor)
  end
end
