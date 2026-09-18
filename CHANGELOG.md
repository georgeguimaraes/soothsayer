# Changelog

## [1.2.1](https://github.com/georgeguimaraes/soothsayer/compare/v1.2.0...v1.2.1) (2026-09-18)


### Bug Fixes

* leave the base slope out of the trend penalty ([53e0f13](https://github.com/georgeguimaraes/soothsayer/commit/53e0f13e8438668f274f6737eec79b806dca2c05))


### Miscellaneous

* Pin Dependabot commit prefix to chore(deps) ([800baab](https://github.com/georgeguimaraes/soothsayer/commit/800baab6919a05c43d3e2e7a0baed73a1e8008af))
* underscores in the retail reference numbers ([f0891cd](https://github.com/georgeguimaraes/soothsayer/commit/f0891cd626ca0412e1a1344bc94f87b210c3dfda))


### Tests

* retail sales, COVID pedestrians and hospital load benchmarks ([3a5b10e](https://github.com/georgeguimaraes/soothsayer/commit/3a5b10edb34b1b3f5c9c23256783ec3db9d02262))

## [1.2.0](https://github.com/georgeguimaraes/soothsayer/compare/v1.1.0...v1.2.0) (2026-09-17)


### Features

* changepoints at dates you know ([2c086b2](https://github.com/georgeguimaraes/soothsayer/commit/2c086b2058834938a95e22a4e0dadf108caad823))
* cross-validation over several cutoffs ([a53d388](https://github.com/georgeguimaraes/soothsayer/commit/a53d3881aaa466c27496742ad8213bd0f8eea265))
* events per series and forecasts for series the model never saw ([c971b63](https://github.com/georgeguimaraes/soothsayer/commit/c971b63c014a1b886c350df8a2c4995b0982973b))
* future_timestamps helper ([65ee5fd](https://github.com/georgeguimaraes/soothsayer/commit/65ee5fd07d86a51802e775e5b1bc3b22a4269cb2))
* logistic growth, a trend that saturates at a cap ([d20619d](https://github.com/georgeguimaraes/soothsayer/commit/d20619dd01eeb1f24fbf78c8312b6e889c0a5c7e))
* recency has an enabled switch like ar and trend ([ae110f8](https://github.com/georgeguimaraes/soothsayer/commit/ae110f8fd3b236fe433bec711a66d315a032e845))


### Bug Fixes

* changepoints sit where NeuralProphet puts them ([151e6e0](https://github.com/georgeguimaraes/soothsayer/commit/151e6e06621d172783304900ad0ece89c6d4e3e4))
* stop referencing Axon.ModelState.t() in specs ([34b9da6](https://github.com/georgeguimaraes/soothsayer/commit/34b9da602b9fe90493523500f48d2454baf8c9bd))


### Documentation

* future_timestamps in the guides, no more semicolons ([f469032](https://github.com/georgeguimaraes/soothsayer/commit/f469032afb2125cd412455057979b872ff203272))
* predict takes a frame for several series, backtest horizon is an AR thing ([d4d7845](https://github.com/georgeguimaraes/soothsayer/commit/d4d784515ed6d8b5d28ffe4fbd2f31478ec02cf2))
* why the trend is detached only at the lag positions ([b36c8e9](https://github.com/georgeguimaraes/soothsayer/commit/b36c8e9b6a7b135c44da7479f80ec18733c062e2))


### Tests

* Prophet as a second reference in the benchmarks ([9e8beed](https://github.com/georgeguimaraes/soothsayer/commit/9e8beeda88e773cf5b040eff25d75f992cd0d12a))
* three more NeuralProphet benchmarks ([84c8538](https://github.com/georgeguimaraes/soothsayer/commit/84c853884184468e92513fb7f2594b932cd72aea))

## [1.1.0](https://github.com/georgeguimaraes/soothsayer/compare/v1.0.0...v1.1.0) (2026-09-17)


### Features

* conformal prediction, intervals calibrated on held out data ([09156ed](https://github.com/georgeguimaraes/soothsayer/commit/09156ed4def6e982cc9b168652a3c354e925059d))
* local trend and seasonality across series, with local regularization ([4c2bba4](https://github.com/georgeguimaraes/soothsayer/commit/4c2bba4af17dc7482dc3364847c305340cb8e369))
* recency weighting, recent rows weigh more in the loss ([f18cfc8](https://github.com/georgeguimaraes/soothsayer/commit/f18cfc8597fda9cf0a7dfe5987585981ccf7ddd0))
* several series in one model ([eb558f9](https://github.com/georgeguimaraes/soothsayer/commit/eb558f98fc2162fc3549fb6fbec69ee5145afcad))


### Documentation

* the 1.1 features in the README, introduction and guides ([51c79c2](https://github.com/georgeguimaraes/soothsayer/commit/51c79c2850a8f8eb34709ee9302564c530332a69))


### Code Refactoring

* fit and predict keep their state per series ([14f3818](https://github.com/georgeguimaraes/soothsayer/commit/14f38186a5a76f879b8c6dda5e7492cd96ac5293))

## [1.0.0](https://github.com/georgeguimaraes/soothsayer/compare/v0.6.3...v1.0.0) (2026-09-16)


### ⚠ BREAKING CHANGES

* country holidays come from dayoff
* event windows are steps_before and steps_after
* Soothsayer.predict/3 returns an Explorer.DataFrame instead of an {n, 1} Nx tensor. Use predictions["yhat"] for the forecast series, or predict_components/3 for tensors.
* **deps:** Require Elixir 1.18, bump credo, dialyxir, ex_doc and align the CI matrix ([#38](https://github.com/georgeguimaraes/soothsayer/issues/38))

### Features

* country holidays come from dayoff ([4111a7a](https://github.com/georgeguimaraes/soothsayer/commit/4111a7ac153c4bbc5499f212456e40b7f6de10bb))
* country holidays through holidefs and yearly recurring events ([4ec24bd](https://github.com/georgeguimaraes/soothsayer/commit/4ec24bda71f77795d55650c4a50ecabe4ab8c8f0))
* custom and conditional seasonalities ([f4a24ee](https://github.com/georgeguimaraes/soothsayer/commit/f4a24ee588fb6981e19b74c6154d08b405f596a8))
* direct multi-step auto-regression with forecast_steps ([bda2d3d](https://github.com/georgeguimaraes/soothsayer/commit/bda2d3dab74fe8216582af235f2f480e53c0191e))
* discontinuous growth ([fd50371](https://github.com/georgeguimaraes/soothsayer/commit/fd503717e283add5d2208aa9223289f4cd1d3f0d))
* event windows are steps_before and steps_after ([f255cc3](https://github.com/georgeguimaraes/soothsayer/commit/f255cc37b245befc3118675643b098ad51b99d84))
* future regressors ([15b0e97](https://github.com/georgeguimaraes/soothsayer/commit/15b0e97efb2b3a5b4aab7cedea2a37025620392b))
* impute and drop missing data the way NeuralProphet does ([e256150](https://github.com/georgeguimaraes/soothsayer/commit/e25615020e8f871764bd8efc85f62e00d4aff7c9))
* lagged regressors ([b4be2ae](https://github.com/georgeguimaraes/soothsayer/commit/b4be2aec39fd99982958432530d453df2c677303))
* learning rate range test, one-cycle schedule and auto epochs as defaults ([388fa58](https://github.com/georgeguimaraes/soothsayer/commit/388fa58ca1e94e8c2846def0c66d7d24ce12519d))
* multiplicative events and regressors ([fe7e031](https://github.com/georgeguimaraes/soothsayer/commit/fe7e0318264c9de559a6e9dbcb336b9e882bfe68))
* multiplicative seasonality and a seed option for reproducible fits ([dad97cd](https://github.com/georgeguimaraes/soothsayer/commit/dad97cdcad0bcc58a02c735830061a9fdd93eb7e))
* naive datetime support, data frequency and daily seasonality ([23aadcf](https://github.com/georgeguimaraes/soothsayer/commit/23aadcf16df449c990a1ca723ca03e033c2da777))
* networks on future regressors and on lagged regressors ([dec669a](https://github.com/georgeguimaraes/soothsayer/commit/dec669a97ea83ec8014cff47ab588ca7f628603c))
* predict returns a dataframe with yhat, quantiles and components ([2ec6033](https://github.com/georgeguimaraes/soothsayer/commit/2ec60337d3c2ceae7a6a3a2c6121aa54c2804b88))
* prediction intervals through quantile regression ([f1522d3](https://github.com/georgeguimaraes/soothsayer/commit/f1522d3b8f6371c1ce88600f50410b4914530ca0))
* regularization for events, regressors and seasonality ([feb0f87](https://github.com/georgeguimaraes/soothsayer/commit/feb0f870d399be36b16a1d858f7b651a2cd69fb9))
* rolling-origin backtest ([b61c7e2](https://github.com/georgeguimaraes/soothsayer/commit/b61c7e21395aa6522fb15dc42b8922e7582650ad))
* segmentwise trend when trend regularization is off ([8510f6d](https://github.com/georgeguimaraes/soothsayer/commit/8510f6ddd830454485a7ab291f6b7db215dbec55))
* stationarized AR lags with per-origin training samples ([eb13d99](https://github.com/georgeguimaraes/soothsayer/commit/eb13d990f11c877908ab4482429491801662fee9))
* Yosemite 5-minute benchmark from NeuralProphet ([6815ac5](https://github.com/georgeguimaraes/soothsayer/commit/6815ac5cc9da165eafa993bf1cc5cedcf4ece855))


### Bug Fixes

* dayoff 0.2, country codes are strings there now ([b7c6aab](https://github.com/georgeguimaraes/soothsayer/commit/b7c6aab78e14810e677e5a6505d43909fa78d0f8))
* forecast AR lags recursively on future dates and accept newer history ([861205e](https://github.com/georgeguimaraes/soothsayer/commit/861205e8418f92d621e312c8b8b8a6cf31a5b2f0))
* keep month-end timestamps on the month-end grid ([799e09b](https://github.com/georgeguimaraes/soothsayer/commit/799e09b3a01d233cb7dadf2429da8dcf5974d09f))
* make predicted components sum to the combined forecast ([7a8c368](https://github.com/georgeguimaraes/soothsayer/commit/7a8c368a47e4521858b810a065139430c0801ce6))
* only the trend has an intercept, so seasonalities carry no level ([a6b65c7](https://github.com/georgeguimaraes/soothsayer/commit/a6b65c70c9e10784b5b55fde029119c9af43f60a))
* read seasonality conditions after missing data handling ([e30c623](https://github.com/georgeguimaraes/soothsayer/commit/e30c623fb0ca0280c972931721ac6dd68579b29e))
* train in shuffled minibatches instead of n_rows full-batch steps per epoch ([0decfba](https://github.com/georgeguimaraes/soothsayer/commit/0decfba171e9be3ba793972d703e4be6574f9d57))
* unknown lags give a NaN forecast instead of a row of zero lags ([c1ee164](https://github.com/georgeguimaraes/soothsayer/commit/c1ee1643230940f6bc03f9bea564718af247925b))


### Miscellaneous

* bump nx and exla to 0.13, explorer to 0.12, test on Elixir 1.18 / OTP 27 ([a242103](https://github.com/georgeguimaraes/soothsayer/commit/a242103586070322f4afaab034a39915f30582f1))
* **deps:** Require Elixir 1.18, bump credo, dialyxir, ex_doc and align the CI matrix ([#38](https://github.com/georgeguimaraes/soothsayer/issues/38)) ([2b88f97](https://github.com/georgeguimaraes/soothsayer/commit/2b88f97440dd1817affb2de8497157be935a76be))
* move to nx and exla 1.0 ([2f58edd](https://github.com/georgeguimaraes/soothsayer/commit/2f58edddbd7f3bdc5543d58ad195a607a69aeb45))


### Documentation

* 1.0 ([d7fc804](https://github.com/georgeguimaraes/soothsayer/commit/d7fc8049cf6f6f9d302ee8bea489c4c1889a544d))
* credit Prophet and the Kaggle author for the benchmark datasets ([1313109](https://github.com/georgeguimaraes/soothsayer/commit/1313109d6847d1b060ffca9e2c809cec29d8bb80))
* describe training samples and zero lags after the stationarized AR change ([65ba792](https://github.com/georgeguimaraes/soothsayer/commit/65ba7925747fd9639309f24295df2888d3a0f32a))
* effect examples in normalized units, networked regressor effects shape ([9cd99bb](https://github.com/georgeguimaraes/soothsayer/commit/9cd99bbe328f217c5d3d75304e9bd7e8be929b64))
* guides and README without the filler ([1ac3c7d](https://github.com/georgeguimaraes/soothsayer/commit/1ac3c7d0431faf5370efaf1678cf8b571c3eecb4))
* list daily seasonality and sub-daily data in the introduction ([1f4fc8c](https://github.com/georgeguimaraes/soothsayer/commit/1f4fc8c8cf4ddd4530a87bbd26741f3e5474ab97))
* note the nx override needed until axon allows 1.0, fix install version ([417cc92](https://github.com/georgeguimaraes/soothsayer/commit/417cc92f5ce7d1bd2621c96e0eb64e7fc4bf7f10))
* predict docs say history is imputed and must be sorted, and drop the gap-free requirement ([c24aafa](https://github.com/georgeguimaraes/soothsayer/commit/c24aafae6dbbbeab9c71582278baf6b027f17976))
* the 1.0 features in the guides, README and introduction ([5514304](https://github.com/georgeguimaraes/soothsayer/commit/5514304c443f09cb574f5e9ef454708a21192d5a))
* years/1 returns an empty range without timestamps ([b2ab311](https://github.com/georgeguimaraes/soothsayer/commit/b2ab311aff2194472333ba46d2e4e362cb429bf9))


### Code Refactoring

* regressors config is a map ([2187b48](https://github.com/georgeguimaraes/soothsayer/commit/2187b48c9a655d0bb6f277b4af486059a8d1fe6c))
* share events and regressors input builders between fit and predict ([2365d90](https://github.com/georgeguimaraes/soothsayer/commit/2365d909b25b6d3d6bf393b74b5e1d1f3a9a29b1))


### Tests

* benchmark Soothsayer against NeuralProphet's model performance datasets ([76d2378](https://github.com/georgeguimaraes/soothsayer/commit/76d2378723e374b3e06c5cc87bf0f631d731afee))


### Build System

* **deps-dev:** bump credo from 1.7.17 to 1.7.18 ([#30](https://github.com/georgeguimaraes/soothsayer/issues/30)) ([2c6bf4b](https://github.com/georgeguimaraes/soothsayer/commit/2c6bf4bbb6fd8522dfad5c247698f066babc2a5a))
* **deps:** bump axon from 0.8.0 to 0.8.1 ([#29](https://github.com/georgeguimaraes/soothsayer/issues/29)) ([401893c](https://github.com/georgeguimaraes/soothsayer/commit/401893c1e74129ed8ec1909b681edc56f9ca35cb))
* **deps:** bump googleapis/release-please-action from 4 to 5 ([#31](https://github.com/georgeguimaraes/soothsayer/issues/31)) ([093d4e0](https://github.com/georgeguimaraes/soothsayer/commit/093d4e05c2c0898aaa82c9f92344ef6e78a76b0d))


### Continuous Integration

* also test on Elixir 1.20 / OTP 29 ([8141dcc](https://github.com/georgeguimaraes/soothsayer/commit/8141dccfb585e8088bdfb6576faf96ca65f46150))


### Performance Improvements

* compile the predict function once and vectorize the time features ([042d942](https://github.com/georgeguimaraes/soothsayer/commit/042d94221912b14e8de56c1606b72c338e8d1354))

## [0.6.3](https://github.com/georgeguimaraes/soothsayer/compare/v0.6.2...v0.6.3) (2026-03-04)


### Bug Fixes

* **ci:** chain hex-publish in release-please workflow ([94740ea](https://github.com/georgeguimaraes/soothsayer/commit/94740ea13d347fe82d05ab6e4c659eefd6ba7b5a))


### Miscellaneous

* remove unused on-release workflow ([a5723fb](https://github.com/georgeguimaraes/soothsayer/commit/a5723fbc5b31e93c3c1057d8f38f37fbde206430))


### Build System

* **deps-dev:** bump credo from 1.7.15 to 1.7.17 ([#28](https://github.com/georgeguimaraes/soothsayer/issues/28)) ([775da92](https://github.com/georgeguimaraes/soothsayer/commit/775da9243dd91415f0096f666ce03036c8215ee4))
* **deps-dev:** bump ex_doc from 0.40.0 to 0.40.1 ([#27](https://github.com/georgeguimaraes/soothsayer/issues/27)) ([e1a9469](https://github.com/georgeguimaraes/soothsayer/commit/e1a946934b434dd95585ed3dabb497c4b418120a))

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
