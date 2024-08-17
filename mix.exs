defmodule Soothsayer.MixProject do
  use Mix.Project

  def project do
    [
      app: :soothsayer,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Soothsayer.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:explorer, "~> 0.9.1"},
      {:nx, "~> 0.7.3"},
      {:axon, "~> 0.6.1"},
      {:exla, "~> 0.7.3"}
    ]
  end
end
