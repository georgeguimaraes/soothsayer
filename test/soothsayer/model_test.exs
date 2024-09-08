defmodule Soothsayer.ModelTest do
  use ExUnit.Case, async: true
  alias Soothsayer.Model
  import Nx, only: [is_tensor: 1]
  import Axon

  test "new/1 creates a new model with the given config" do
    config = %{
      trend: %{enabled: true},
      seasonality: %{
        yearly: %{enabled: true, fourier_terms: 4},
        weekly: %{enabled: true, fourier_terms: 2}
      },
      learning_rate: 0.01,
      epochs: 100
    }
    model = Model.new(config)
    assert %Model{} = model
    assert model.config == config
    assert is_struct(model.network, Axon)
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
    inputs = Axon.get_inputs(network)
    assert Map.has_key?(inputs, "trend")
    assert Map.has_key?(inputs, "yearly")
    assert Map.has_key?(inputs, "weekly")

    # Check network structure
    assert map_size(Axon.nodes(network)) > 0

    # Check output shape
    assert Axon.get_output_shape(network, %{
      "trend" => {1, 1},
      "yearly" => {1, 8},
      "weekly" => {1, 4}
    }) == {1, 1}
  end

  test "fit/4 trains the model" do
    config = %{
      trend: %{enabled: true},
      seasonality: %{
        yearly: %{enabled: true, fourier_terms: 4},
        weekly: %{enabled: true, fourier_terms: 2}
      },
      learning_rate: 0.01,
      epochs: 1
    }
    model = Model.new(config)
    
    x = %{
      "trend" => Nx.tensor([[1.0], [2.0], [3.0]]),
      "yearly" => Nx.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
                             [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                             [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]),
      "weekly" => Nx.tensor([[0.1, 0.2, 0.3, 0.4], 
                             [0.2, 0.3, 0.4, 0.5],
                             [0.3, 0.4, 0.5, 0.6]])
    }
    y = Nx.tensor([[1.0], [2.0], [3.0]])

    trained_model = Model.fit(model, x, y, 1)
    assert is_struct(trained_model, Model)
    assert trained_model.params != nil
  end

  test "predict/2 makes predictions" do
    config = %{
      trend: %{enabled: true},
      seasonality: %{
        yearly: %{enabled: true, fourier_terms: 4},
        weekly: %{enabled: true, fourier_terms: 2}
      },
      learning_rate: 0.01,
      epochs: 1
    }
    model = Model.new(config)
    
    x = %{
      "trend" => Nx.tensor([[1.0], [2.0], [3.0]]),
      "yearly" => Nx.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
                             [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                             [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]),
      "weekly" => Nx.tensor([[0.1, 0.2, 0.3, 0.4], 
                             [0.2, 0.3, 0.4, 0.5],
                             [0.3, 0.4, 0.5, 0.6]])
    }
    y = Nx.tensor([[1.0], [2.0], [3.0]])

    trained_model = Model.fit(model, x, y, 1)
    predictions = Model.predict(trained_model, x)
    
    assert is_map(predictions)
    assert Map.has_key?(predictions, :combined)
    assert is_tensor(predictions.combined)
    assert Nx.shape(predictions.combined) == {3, 1}
  end
end
