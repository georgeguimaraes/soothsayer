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
    assert is_map(network)
    # Add more specific assertions about the network structure if needed
  end

  # Add more tests for fit/4 and predict/2 methods
  # These tests might require more setup and mocking of Axon functionality
end
