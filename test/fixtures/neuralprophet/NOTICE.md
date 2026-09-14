# NeuralProphet benchmark datasets

These CSV files come from the NeuralProphet test suite, `tests/test-data/` in
https://github.com/ourownstory/neural_prophet at commit `5e6b23145473`
(2024-09-13). NeuralProphet is MIT licensed, copyright (c) 2020 Oskar Triebe.

| File | Origin | Notes |
|------|--------|-------|
| `wp_log_peyton_manning.csv` | `wp_log_peyton_manning.csv` | Daily log page views, 2905 rows |
| `air_passengers.csv` | `air_passengers.csv` | Monthly passengers, 144 rows, line endings converted from CR to LF |
| `energy_price_daily.csv` | `tutorial04_kaggle_energy_daily_temperature.csv` | Daily energy price with a temperature column, 1461 rows |

They back `test/soothsayer/neuralprophet_benchmark_test.exs`, which fits
Soothsayer on the same splits NeuralProphet's `tests/test_model_performance.py`
uses and reports the metrics side by side.
