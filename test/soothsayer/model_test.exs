defmodule Soothsayer.ModelTest do
  use ExUnit.Case, async: true
  alias Soothsayer.Model

  test "new/1 creates a new model with the given config" do
    config = %{
      trend: %{enabled: true},
      seasonality: %{
        yearly: %{enabled: true, fourier_terms: 4},
        weekly: %{enabled: true, fourier_terms: 2}
      },
      learning_rate: 0.01
    }
    model = Model.new(config)
    assert %Model{} = model
    assert model.config == config
    assert is_map(model.network)
  end

  test "build_network/1 creates a network based on the config" do
    config = %{
      trend: %{enabled: true},
      seasonality: %{
        yearly: %{enabled: true, fourier_terms: 4},
        weekly: %{enabled: true, fourier_terms: 2}
      }
    }
    network = Model.build_network(config)
    assert is_struct(network, Axon)

    # Check input shapes
    assert Axon.get_input_shape(network, "trend") == {nil, 1}
    assert Axon.get_input_shape(network, "yearly") == {nil, 8}  # 2 * 4 fourier terms
    assert Axon.get_input_shape(network, "weekly") == {nil, 4}  # 2 * 2 fourier terms

    # Check network structure
    layers = Axon.get_layers(network)
    assert length(layers) > 0

    # Check for trend layer
    trend_layer = Enum.find(layers, &(&1.name == "dense"))
    assert trend_layer != nil
    assert trend_layer.op == :dense
    assert trend_layer.opts[:units] == 1
    assert trend_layer.opts[:activation] == :linear

    # Check for the final addition layer
    add_layer = List.last(layers)
    assert add_layer.op == :add

    # Check output shape
    assert Axon.get_output_shape(network) == {nil, 1}
  end

  # Add more tests for fit/4 and predict/2 methods
  # These tests might require more setup and mocking of Axon functionality
end
