defmodule Soothsayer do
  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Model

  def new do
    Model.new()
  end

  def fit(%Model{} = model, %DataFrame{} = data) do
    x = data["x"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)
    y = data["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)

    {init_fn, predict_fn} = Axon.build(model.nn_model)
    initial_params = init_fn.(Nx.template({Nx.size(x), 1}, {:f, 32}), %{})

    train_data =
      Stream.repeatedly(fn ->
        {x, y}
      end)

    trained_params =
      model.nn_model
      |> Axon.Loop.trainer(:mean_squared_error, :adam)
      |> Axon.Loop.run(train_data, initial_params, epochs: 100, iterations: Nx.size(x))

    %{model | nn_model: model.nn_model, params: trained_params}
  end

  def predict(%Model{nn_model: model, params: params}, %Series{} = x) do
    x_tensor = x |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)
    {_init_fn, predict_fn} = Axon.build(model)
    predict_fn.(params, x_tensor)
  end
end
