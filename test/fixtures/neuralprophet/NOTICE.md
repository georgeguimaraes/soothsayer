# Benchmark datasets

These CSV files were taken from the NeuralProphet test suite, `tests/test-data/`
in https://github.com/ourownstory/neural_prophet at commit `5e6b23145473`
(2024-09-13), so that `test/soothsayer/neuralprophet_benchmark_test.exs` can fit
Soothsayer on the same splits NeuralProphet's `tests/test_model_performance.py`
uses and report the metrics side by side. NeuralProphet is MIT licensed,
copyright (c) 2020 Oskar Triebe, but the data itself comes from elsewhere:

| File | Origin | Copyright and license |
|------|--------|-----------------------|
| `wp_log_peyton_manning.csv` | Byte-identical to `examples/example_wp_log_peyton_manning.csv` in [Prophet](https://github.com/facebook/prophet). Daily log of Wikipedia page views for Peyton Manning, 2905 rows | Copyright (c) Facebook, Inc. and its affiliates, MIT License |
| `yosemite_temps.csv` | Byte-identical to `examples/example_yosemite_temps.csv` in Prophet. Temperature every 5 minutes, 18721 rows over 65 days, 12 missing readings | Copyright (c) Facebook, Inc. and its affiliates, MIT License |
| `air_passengers.csv` | Byte-identical to `examples/example_air_passengers.csv` in Prophet, apart from line endings converted from CR to LF. The classic Box and Jenkins monthly airline passengers series (1949 to 1960), 144 rows | Public domain data, redistributed by Prophet under the MIT License |
| `energy_price_daily.csv` | NeuralProphet's `tutorial04_kaggle_energy_daily_temperature.csv`, a prepared daily cut of the Kaggle dataset [Hourly energy demand generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) by Nicholas Jhana (data from ENTSO-E, REE and OpenWeather). Daily energy price with a temperature column, 1461 rows | CC0 1.0 Public Domain |

Prophet's MIT License: https://github.com/facebook/prophet/blob/main/LICENSE
