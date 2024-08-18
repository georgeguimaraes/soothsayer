defmodule Soothsayer do
  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Model
  alias Soothsayer.Preprocessor

  def new(config \\ %{}) do
    Model.new(config)
  end

  def fit(%Model{} = model, %DataFrame{} = data) do
    processed_data = Preprocessor.prepare_data(data, "y", "ds", model.seasonality_config)

    x_columns = processed_data.names -- ["y", "ds"]

    x =
      processed_data[x_columns]
      |> DataFrame.to_series()
      |> Map.values()
      |> Enum.map(&Series.to_tensor/1)

    x =
      if Enum.empty?(x) do
        # If no features (trend and seasonality disabled), use a dummy feature
        [Nx.tensor(List.duplicate(1, Series.size(processed_data["y"])))]
      else
        x
      end

    x = Nx.stack(x, axis: 1) |> Nx.as_type({:f, 32})

    y = processed_data["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)

    # Normalize input
    {x_normalized, x_mean, x_std} = normalize(x)
    {y_normalized, y_mean, y_std} = normalize(y)

    nn_model = Model.build(model)
    {init_fn, predict_fn} = Axon.build(nn_model)
    initial_params = init_fn.(x_normalized, %{})

    train_data =
      Stream.repeatedly(fn ->
        {x_normalized, y_normalized}
      end)

    trained_params =
      nn_model
      |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adam())
      |> Axon.Loop.run(train_data, initial_params,
        epochs: 10,
        iterations: Nx.shape(x) |> elem(0),
        compiler: EXLA
      )

    %{
      model
      | nn_model: nn_model,
        params: trained_params,
        trend_config:
          Map.put(model.trend_config, :normalization, %{
            x_mean: x_mean,
            x_std: x_std,
            y_mean: y_mean,
            y_std: y_std
          })
    }
  end

  def predict(
        %Model{
          nn_model: model,
          params: params,
          trend_config: trend_config,
          seasonality_config: seasonality_config
        },
        %Series{} = x
      ) do
    processed_x =
      Preprocessor.prepare_data(DataFrame.new(%{"ds" => x}), "y", "ds", seasonality_config)

    x_columns = processed_x.names -- ["y", "ds"]

    x_tensor =
      processed_x[x_columns]
      |> DataFrame.to_series()
      |> Map.values()
      |> Enum.map(&Series.to_tensor/1)

    x_tensor =
      if Enum.empty?(x_tensor) do
        # If no features (trend and seasonality disabled), use a dummy feature
        [Nx.tensor(List.duplicate(1, Series.size(x)))]
      else
        x_tensor
      end

    x_tensor = Nx.stack(x_tensor, axis: 1) |> Nx.as_type({:f, 32})

    x_normalized =
      Nx.subtract(x_tensor, trend_config.normalization.x_mean)
      |> Nx.divide(trend_config.normalization.x_std)

    {_init_fn, predict_fn} = Axon.build(model)
    y_normalized = predict_fn.(params, x_normalized)

    y_denormalized =
      Nx.multiply(y_normalized, trend_config.normalization.y_std)
      |> Nx.add(trend_config.normalization.y_mean)

    Nx.squeeze(y_denormalized)
  end

  defp normalize(tensor) do
    mean = Nx.mean(tensor, axes: [0])
    std = Nx.standard_deviation(tensor, axes: [0])
    std = Nx.select(Nx.equal(std, 0), Nx.tensor(1), std)
    {Nx.divide(Nx.subtract(tensor, mean), std), mean, std}
  end
end
