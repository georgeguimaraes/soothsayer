# Changelog

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
