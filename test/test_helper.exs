Nx.global_default_backend(EXLA.Backend)
ExUnit.start(timeout: 600_000, exclude: [:benchmark], capture_log: true)
