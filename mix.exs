defmodule Soothsayer.MixProject do
  use Mix.Project

  @version "0.2.0"
  @source_url "https://github.com/georgeguimaraes/soothsayer"

  def project do
    [
      app: :soothsayer,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {Soothsayer.Application, []}
    ]
  end

  defp deps do
    [
      {:explorer, "~> 0.9.1"},
      {:nx, "~> 0.7.3"},
      {:axon, "~> 0.6.1"},
      {:exla, "~> 0.7.3"}
    ]
  end

  defp package do
    [
      description:
        "Soothsayer is an Elixir library for time series forecasting, inspired by Facebook's Prophet and NeuralProphet.",
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url
      }
    ]
  end

  defp docs do
    [
      main: "readme",
      source_url: @source_url,
      source_ref: "v#{@version}",
      extras: [
        "README.md"
      ]
    ]
  end
end
