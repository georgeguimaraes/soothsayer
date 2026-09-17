defmodule Soothsayer.MixProject do
  use Mix.Project

  @version "1.2.0"
  @source_url "https://github.com/georgeguimaraes/soothsayer"

  def project do
    [
      app: :soothsayer,
      name: "Soothsayer",
      version: @version,
      elixir: "~> 1.18",
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
      {:explorer, "~> 0.12.0"},
      # Hex refuses packages with an overridden dependency, and axon 0.8.x still declares
      # nx ~> 0.10, so the range lets the resolver pick 0.13 today and 1.0 once axon moves.
      {:nx, "~> 0.13 or ~> 1.0"},
      {:axon, "~> 0.8.0"},
      {:exla, "~> 0.13 or ~> 1.0"},
      # Country holidays for holidays: %{countries: [...]}.
      {:dayoff, "~> 0.2"},
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
      before_closing_body_tag: fn _format ->
        """
        <footer style="padding: 1rem 0; margin-top: 2rem; border-top: 1px solid #e1e4e8; font-size: 0.875rem; color: #586069;">
          Copyright 2024 George Guimarães. Licensed under Apache-2.0.
        </footer>
        """
      end,
      deps: [
        axon: "https://hexdocs.pm/axon/",
        nx: "https://hexdocs.pm/nx/",
        explorer: "https://hexdocs.pm/explorer/",
        dayoff: "https://hexdocs.pm/dayoff/"
      ],
      extras: [
        "README.md",
        "guides/introduction.md",
        "guides/basics.md",
        "guides/trends.md",
        "guides/seasonality.md",
        "guides/autoregression.md",
        "guides/events.md",
        "guides/regressors.md",
        "guides/missing_data.md",
        "guides/uncertainty.md",
        "guides/series.md"
      ],
      groups_for_extras: [
        Tutorials: [
          "guides/introduction.md",
          "guides/basics.md",
          "guides/trends.md",
          "guides/seasonality.md",
          "guides/autoregression.md",
          "guides/events.md",
          "guides/regressors.md",
          "guides/missing_data.md",
          "guides/uncertainty.md",
          "guides/series.md"
        ]
      ]
    ]
  end
end
