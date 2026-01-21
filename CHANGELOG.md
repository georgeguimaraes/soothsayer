# Changelog

## [0.6.2](https://github.com/georgeguimaraes/soothsayer/compare/v0.6.1...v0.6.2) (2026-01-21)


### Miscellaneous

* add dependabot for daily updates ([0fbf931](https://github.com/georgeguimaraes/soothsayer/commit/0fbf931ae9b7a8b2abb5fb0b6a569cce2a9c012c))
* remove beads ([9d93d22](https://github.com/georgeguimaraes/soothsayer/commit/9d93d2289a403d9ccd885341a05069d5774b0983))
* Simplify release-please workflow ([cc60f4e](https://github.com/georgeguimaraes/soothsayer/commit/cc60f4ea86dd7be871b147b713e1d6b47816a163))


### Documentation

* Add GPU memory configuration guidance ([ede24ff](https://github.com/georgeguimaraes/soothsayer/commit/ede24ff388bd43f4b4bfea7d9bda7492eb94baca))


### Code Refactoring

* **ci:** use release-please for GitHub releases ([12b56c7](https://github.com/georgeguimaraes/soothsayer/commit/12b56c788b7c2ced42bd45b0e5ebec96d1a00c50))


### Build System

* **deps-dev:** bump credo from 1.7.14 to 1.7.15 ([#22](https://github.com/georgeguimaraes/soothsayer/issues/22)) ([b86c3de](https://github.com/georgeguimaraes/soothsayer/commit/b86c3de7e58d524cbeedd2bbf836814af13009d2))
* **deps-dev:** bump ex_doc from 0.39.3 to 0.40.0 ([#24](https://github.com/georgeguimaraes/soothsayer/issues/24)) ([8e59232](https://github.com/georgeguimaraes/soothsayer/commit/8e5923253657d6c016d3ee44392e081864452a49))
* **deps:** bump actions/cache from 3 to 5 ([#20](https://github.com/georgeguimaraes/soothsayer/issues/20)) ([10add51](https://github.com/georgeguimaraes/soothsayer/commit/10add51d3c84076dc6a1bb517ad99fb637ed85ab))
* **deps:** bump actions/checkout from 4 to 6 ([#21](https://github.com/georgeguimaraes/soothsayer/issues/21)) ([c8e45aa](https://github.com/georgeguimaraes/soothsayer/commit/c8e45aa281bb2d8a1cb9945de167dbfa7f43a371))
* **deps:** bump amannn/action-semantic-pull-request from 5 to 6 ([#19](https://github.com/georgeguimaraes/soothsayer/issues/19)) ([b10c0eb](https://github.com/georgeguimaraes/soothsayer/commit/b10c0ebc564932546cc1865d367466491eb467b7))


### Continuous Integration

* Add create-release workflow ([9e15192](https://github.com/georgeguimaraes/soothsayer/commit/9e15192c8bb9c9f15b8a96b51c79feab3ce9b6eb))
* use shared workflows from georgeguimaraes/workflows ([d6e5d34](https://github.com/georgeguimaraes/soothsayer/commit/d6e5d34709f957e878bac717772ab98f9effff10))

## [0.6.1](https://github.com/georgeguimaraes/soothsayer/compare/v0.6.0...v0.6.1) (2025-12-28)


### Bug Fixes

* Align dataframe columns for concat in forecast example ([ef0d01c](https://github.com/georgeguimaraes/soothsayer/commit/ef0d01c8dd400e14b9a2b26c7599951375431ff8))

## [0.6.0](https://github.com/georgeguimaraes/soothsayer/compare/v0.5.0...v0.6.0) (2025-12-20)


### Features

* Add auto-regression (AR) support ([#9](https://github.com/georgeguimaraes/soothsayer/issues/9)) ([4a58d36](https://github.com/georgeguimaraes/soothsayer/commit/4a58d36230d87fc35b7c844ad386e5f0deeed202))
* Add display_network/1 for Axon.Display.as_graph compatibility ([1f9ead7](https://github.com/georgeguimaraes/soothsayer/commit/1f9ead7e6e4061b41dff639a3684fc03f111e34e))
* Add events and holidays support ([c4cecba](https://github.com/georgeguimaraes/soothsayer/commit/c4cecbafd0ab784d89a30c406dd2240d0f7fb1d7))
* Add get_ar_weights/1 for AR layer weight inspection ([#11](https://github.com/georgeguimaraes/soothsayer/issues/11)) ([4ad2db6](https://github.com/georgeguimaraes/soothsayer/commit/4ad2db65ae2c8e7976c7d9022e910505a2366876))
* Add piecewise linear trend with changepoint detection ([#12](https://github.com/georgeguimaraes/soothsayer/issues/12)) ([d39d3a6](https://github.com/georgeguimaraes/soothsayer/commit/d39d3a6642c442503cb8e9e003721f97d33a778a))
* Add typespecs notation to each public function in lib/ ([e88ae7f](https://github.com/georgeguimaraes/soothsayer/commit/e88ae7f235faba3b9470d0889eb1e9e42da24772))
* Change seasonality component activation to ReLU ([f923ca4](https://github.com/georgeguimaraes/soothsayer/commit/f923ca48c8c3f3bfcccc732f9372d2442dd19cc9))
* Output components with combined value ([a8e04b2](https://github.com/georgeguimaraes/soothsayer/commit/a8e04b2d78fb8333c6fb95664f287bdd564d390e))
* Update model training to use Huber loss and higher learning rate ([52a56ce](https://github.com/georgeguimaraes/soothsayer/commit/52a56cef81b311ba3687ecfca44019743e9cc01c))


### Bug Fixes

* Add docs and typespecs ([#4](https://github.com/georgeguimaraes/soothsayer/issues/4)) ([e88ae7f](https://github.com/georgeguimaraes/soothsayer/commit/e88ae7f235faba3b9470d0889eb1e9e42da24772))
* Decrease the learning rate in the Soothsayer model ([d947782](https://github.com/georgeguimaraes/soothsayer/commit/d947782b9c58cd45cd7cae5c67001e11042e3eea))
* enable Hex.pm publishing in release workflow ([#13](https://github.com/georgeguimaraes/soothsayer/issues/13)) ([b7a33ce](https://github.com/georgeguimaraes/soothsayer/commit/b7a33ce5fa8a86c35cc10243eab03034cd585b02))
* remove duplicated test action yml key ([be0d144](https://github.com/georgeguimaraes/soothsayer/commit/be0d1444bbe55b3b5e9d2777db20306efa869adb))
* Replace invalid loss function with mean_squared_error ([2f25af3](https://github.com/georgeguimaraes/soothsayer/commit/2f25af39b5a68d4fecb6b5ae7d50bf2ad75438b0))
* Resolve ExDoc warnings for type refs and file links ([303a8a0](https://github.com/georgeguimaraes/soothsayer/commit/303a8a09814b8f89cf0a2df9b1434391cd792056))
* update version ([cc65eb4](https://github.com/georgeguimaraes/soothsayer/commit/cc65eb499aa94799f2ae7b4fb0a1e501ca5d9944))

## [0.5.0](https://github.com/georgeguimaraes/soothsayer/compare/v0.4.1...v0.5.0) (2025-12-20)


### Features

* Add display_network/1 for Axon.Display.as_graph compatibility ([1f9ead7](https://github.com/georgeguimaraes/soothsayer/commit/1f9ead7e6e4061b41dff639a3684fc03f111e34e))
* Add events and holidays support ([c4cecba](https://github.com/georgeguimaraes/soothsayer/commit/c4cecbafd0ab784d89a30c406dd2240d0f7fb1d7))

## [0.4.1](https://github.com/georgeguimaraes/soothsayer/compare/v0.4.0...v0.4.1) (2025-12-19)


### Bug Fixes

* enable Hex.pm publishing in release workflow ([#13](https://github.com/georgeguimaraes/soothsayer/issues/13)) ([b7a33ce](https://github.com/georgeguimaraes/soothsayer/commit/b7a33ce5fa8a86c35cc10243eab03034cd585b02))

## [0.4.0](https://github.com/georgeguimaraes/soothsayer/compare/v0.3.1...v0.4.0) (2025-12-19)


### Features

* Add auto-regression (AR) support ([#9](https://github.com/georgeguimaraes/soothsayer/issues/9)) ([4a58d36](https://github.com/georgeguimaraes/soothsayer/commit/4a58d36230d87fc35b7c844ad386e5f0deeed202))
* Add get_ar_weights/1 for AR layer weight inspection ([#11](https://github.com/georgeguimaraes/soothsayer/issues/11)) ([4ad2db6](https://github.com/georgeguimaraes/soothsayer/commit/4ad2db65ae2c8e7976c7d9022e910505a2366876))
* Add piecewise linear trend with changepoint detection ([#12](https://github.com/georgeguimaraes/soothsayer/issues/12)) ([d39d3a6](https://github.com/georgeguimaraes/soothsayer/commit/d39d3a6642c442503cb8e9e003721f97d33a778a))


### Bug Fixes

* Resolve ExDoc warnings for type refs and file links ([303a8a0](https://github.com/georgeguimaraes/soothsayer/commit/303a8a09814b8f89cf0a2df9b1434391cd792056))

## [0.3.1](https://github.com/georgeguimaraes/soothsayer/compare/v0.3.0...v0.3.1) (2024-09-10)


### Bug Fixes

* Add docs and typespecs ([#4](https://github.com/georgeguimaraes/soothsayer/issues/4)) ([e88ae7f](https://github.com/georgeguimaraes/soothsayer/commit/e88ae7f235faba3b9470d0889eb1e9e42da24772))

## [0.3.0](https://github.com/georgeguimaraes/soothsayer/compare/v0.2.0...v0.3.0) (2024-09-08)


### Features

* Output components with combined value ([a8e04b2](https://github.com/georgeguimaraes/soothsayer/commit/a8e04b2d78fb8333c6fb95664f287bdd564d390e))


### Bug Fixes

* remove duplicated test action yml key ([be0d144](https://github.com/georgeguimaraes/soothsayer/commit/be0d1444bbe55b3b5e9d2777db20306efa869adb))
