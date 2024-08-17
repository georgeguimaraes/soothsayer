defmodule Soothsayer.Model do
  @moduledoc """
  Represents a Soothsayer model with its parameters and state.
  """

  defstruct [:params, :state]

  @type t :: %__MODULE__{
          params: map(),
          state: map()
        }
end
