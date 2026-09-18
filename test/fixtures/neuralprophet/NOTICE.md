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
| `wp_log_R_outliers.csv` | `examples/example_wp_log_R_outliers1.csv` in Prophet, rows sorted by `ds` and the quotes stripped. Daily log of Wikipedia page views for the R programming language page, 2008 to 2015, with the outlier spikes Prophet's docs use, 2863 rows | Copyright (c) Facebook, Inc. and its affiliates, MIT License |
| `births_us.csv` | `datasets/births_us.csv` in [neuralprophet-data](https://github.com/ourownstory/neuralprophet-data). Daily number of births in the US, 1969 to 1988, 7305 rows, the National Center for Health Statistics series used in Gelman et al.'s birthdays example and in NeuralProphet's tutorials | US federal government data, public domain |
| `pedestrians_panel.csv` | `examples/example_pedestrians_multivariate.csv` in Prophet, reshaped from one column per location to one row per location and hour with an `id` column. Hourly pedestrian counts at two Melbourne locations over April 2023, 1440 rows | Copyright (c) Facebook, Inc. and its affiliates, MIT License; the counts come from the City of Melbourne's open data (CC BY 4.0) |
| `retail_sales.csv` | Byte-identical to `examples/example_retail_sales.csv` in Prophet. Monthly US retail sales (the Census Bureau's advance monthly sales series), 1992 to 2016, 293 rows | Copyright (c) Facebook, Inc. and its affiliates, MIT License; the counts are US federal government data, public domain |
| `pedestrians_covid.csv` | Byte-identical to `examples/example_pedestrians_covid.csv` in Prophet. Daily pedestrian counts at one Melbourne location, June 2017 to June 2021, through the COVID lockdowns, 1490 rows | Copyright (c) Facebook, Inc. and its affiliates, MIT License; the counts come from the City of Melbourne's open data (CC BY 4.0) |
| `hospital_load.csv` | `datasets/energy/SF_hospital_load.csv` in [neuralprophet-data](https://github.com/ourownstory/neuralprophet-data). Hourly electricity load of a San Francisco hospital over 2015, 8760 rows, from OpenEI's commercial building load profiles (US Department of Energy) | US federal government data, public domain; neuralprophet-data is MIT licensed |
| `energy_price_daily.csv` | NeuralProphet's `tutorial04_kaggle_energy_daily_temperature.csv`, a prepared daily cut of the Kaggle dataset [Hourly energy demand generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) by Nicholas Jhana (data from ENTSO-E, REE and OpenWeather). Daily energy price with a temperature column, 1461 rows | CC0 1.0 Public Domain |

Prophet's MIT License: https://github.com/facebook/prophet/blob/main/LICENSE
