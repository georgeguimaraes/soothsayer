defmodule Soothsayer.MixProject do
  use Mix.Project

  @version "0.4.1"
  @source_url "https://github.com/georgeguimaraes/soothsayer"

  def project do
    [
      app: :soothsayer,
      name: "Soothsayer",
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      docs: docs(),
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:explorer, "~> 0.11.1"},
      {:nx, "~> 0.10.0"},
      {:axon, "~> 0.8.0"},
      {:exla, "~> 0.10.0"},
      {:ex_doc, ">= 0.0.0", only: :docs},
      {:dialyxir, "~> 1.0", only: [:dev], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end

  defp package do
    [
      maintainers: ["George Guimarães"],
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
      authors: ["George Guimarães"],
      source_url: @source_url,
      source_ref: "v#{@version}",
      deps: [
        axon: "https://hexdocs.pm/axon/",
        nx: "https://hexdocs.pm/nx/",
        explorer: "https://hexdocs.pm/explorer/"
      ],
      extras: [
        "README.md",
        "guides/introduction.md",
        "guides/basics.md",
        "guides/trends.md",
        "guides/seasonality.md",
        "guides/autoregression.md",
        "guides/events.md"
      ],
      groups_for_extras: [
        Tutorials: [
          "guides/introduction.md",
          "guides/basics.md",
          "guides/trends.md",
          "guides/seasonality.md",
          "guides/autoregression.md",
          "guides/events.md"
        ]
      ]
    ]
  end
end
