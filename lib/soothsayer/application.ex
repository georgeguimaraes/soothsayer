defmodule Soothsayer.Application do
  use Application

  def start(_type, _args) do
    Nx.global_default_backend(EXLA.Backend)

    children = [
      # Add any necessary children here
    ]

    opts = [strategy: :one_for_one, name: Soothsayer.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
