defmodule SoothsayerTest do
  use ExUnit.Case
  doctest Soothsayer

  test "greets the world" do
    assert Soothsayer.hello() == :world
  end
end
