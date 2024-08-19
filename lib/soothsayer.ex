defmodule Soothsayer do
  alias Explorer.DataFrame
  alias Explorer.Series
  alias Soothsayer.Model
  alias Soothsayer.Preprocessor

  def new(config \\ %{}) do
    default_config = %{
      trend_config: %{enabled: true},
      seasonality_config: %{
        yearly: %{enabled: true, fourier_terms: 6},
        weekly: %{enabled: true, fourier_terms: 3}
      },
      # Default to 100 epochs
      epochs: 100
    }

    merged_config = Map.merge(default_config, config)
    Model.new(merged_config)
  end

  def fit(%Model{} = model, %DataFrame{} = data) do
    processed_data = Preprocessor.prepare_data(data, "y", "ds", model.seasonality_config)

    y = processed_data["y"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)
    {y_normalized, y_mean, y_std} = normalize(y)

    # Fit trend
    {trend_params, trend_norm} =
      if model.trend_config.enabled do
        trend_input =
          processed_data["ds"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)

        {trend_input_normalized, trend_mean, trend_std} = normalize(trend_input)

        {_, params} =
          Model.fit_model(model.trend_model, trend_input_normalized, y_normalized, model.epochs)

        {params, %{input_mean: trend_mean, input_std: trend_std}}
      else
        {nil, nil}
      end

    # Fit yearly seasonality
    {yearly_params, yearly_norm} =
      if model.seasonality_config.yearly.enabled do
        yearly_input = get_seasonality_input(processed_data, :yearly)
        {yearly_input_normalized, yearly_mean, yearly_std} = normalize(yearly_input)

        {_, params} =
          Model.fit_model(
            model.yearly_seasonality_model,
            yearly_input_normalized,
            y_normalized,
            model.epochs
          )

        {params, %{input_mean: yearly_mean, input_std: yearly_std}}
      else
        {nil, nil}
      end

    # Fit weekly seasonality
    {weekly_params, weekly_norm} =
      if model.seasonality_config.weekly.enabled do
        weekly_input = get_seasonality_input(processed_data, :weekly)
        {weekly_input_normalized, weekly_mean, weekly_std} = normalize(weekly_input)

        {_, params} =
          Model.fit_model(
            model.weekly_seasonality_model,
            weekly_input_normalized,
            y_normalized,
            model.epochs
          )

        {params, %{input_mean: weekly_mean, input_std: weekly_std}}
      else
        {nil, nil}
      end

    %{
      model
      | params: %{
          trend: trend_params,
          yearly: yearly_params,
          weekly: weekly_params
        },
        trend_config: Map.put(model.trend_config, :normalization, trend_norm),
        seasonality_config:
          Map.put(model.seasonality_config, :normalization, %{
            yearly: yearly_norm,
            weekly: weekly_norm
          }),
        y_normalization: %{mean: y_mean, std: y_std}
    }
  end

  def predict(%Model{} = model, %Series{} = x) do
    components = predict_components(model, x)
    combined = combine_components(components)
    denormalize(Nx.tensor(combined), model.y_normalization) |> Nx.to_flat_list()
  end

  defp combine_components(components) do
    Enum.zip_with(
      [components.trend, components.yearly_seasonality, components.weekly_seasonality],
      fn values -> Enum.sum(values) end
    )
  end

  def predict_components(%Model{} = model, %Series{} = x) do
    processed_x =
      Preprocessor.prepare_data(DataFrame.new(%{"ds" => x}), nil, "ds", model.seasonality_config)

    x_size = Series.size(x)

    # Predict trend
    trend =
      if model.trend_config.enabled do
        trend_input =
          processed_x["ds"] |> Series.to_tensor() |> Nx.as_type({:f, 32}) |> Nx.new_axis(-1)

        trend_input_normalized =
          normalize_with_params(trend_input, model.trend_config.normalization)

        pred = Model.predict(model.trend_model, model.params.trend, trend_input_normalized)
        pred |> Nx.to_flat_list()
      else
        List.duplicate(0, x_size)
      end

    # Predict yearly seasonality
    yearly =
      if model.seasonality_config.yearly.enabled do
        yearly_input = get_seasonality_input(processed_x, :yearly)

        yearly_input_normalized =
          normalize_with_params(yearly_input, model.seasonality_config.normalization.yearly)

        pred =
          Model.predict(
            model.yearly_seasonality_model,
            model.params.yearly,
            yearly_input_normalized
          )

        pred |> Nx.to_flat_list()
      else
        List.duplicate(0, x_size)
      end

    # Predict weekly seasonality
    weekly =
      if model.seasonality_config.weekly.enabled do
        weekly_input = get_seasonality_input(processed_x, :weekly)

        weekly_input_normalized =
          normalize_with_params(weekly_input, model.seasonality_config.normalization.weekly)

        pred =
          Model.predict(
            model.weekly_seasonality_model,
            model.params.weekly,
            weekly_input_normalized
          )

        pred |> Nx.to_flat_list()
      else
        List.duplicate(0, x_size)
      end

    %{
      trend: trend,
      yearly_seasonality: yearly,
      weekly_seasonality: weekly
    }
  end

  defp get_seasonality_input(data, seasonality) do
    columns = data.names |> Enum.filter(&String.starts_with?(&1, Atom.to_string(seasonality)))

    data[columns]
    |> DataFrame.to_series()
    |> Map.values()
    |> Enum.map(&Series.to_tensor/1)
    |> Nx.stack(axis: 1)
    |> Nx.as_type({:f, 32})
  end

  defp normalize(tensor) do
    mean = Nx.mean(tensor, axes: [0])
    std = Nx.standard_deviation(tensor, axes: [0])
    std = Nx.select(Nx.equal(std, 0), Nx.tensor(1), std)
    {Nx.divide(Nx.subtract(tensor, mean), std), mean, std}
  end

  defp normalize_with_params(tensor, %{input_mean: mean, input_std: std}) do
    Nx.divide(Nx.subtract(tensor, mean), std)
  end

  defp denormalize(tensor, %{mean: mean, std: std}) do
    Nx.multiply(tensor, std) |> Nx.add(mean)
  end
end
