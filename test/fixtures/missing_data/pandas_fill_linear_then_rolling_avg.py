# Reference outputs for Soothsayer.MissingData.impute/3, from NeuralProphet's
# fill_linear_then_rolling_avg (neuralprophet/df_utils.py) run with pandas:
#   uv run --with pandas --with numpy python pandas_fill_linear_then_rolling_avg.py
import json
import numpy as np
import pandas as pd

def fill_linear_then_rolling_avg(series, limit_linear, rolling):
    series = pd.to_numeric(series)
    series = series.interpolate(method="linear", limit=limit_linear, limit_direction="both")
    is_na = pd.isna(series)
    rolling_avg = series.rolling(rolling + 2 * limit_linear, min_periods=2 * limit_linear, center=True).mean()
    series.loc[is_na] = rolling_avg[is_na]
    return series

def case(name, before, gap, after, linear, rolling, lead=0):
    values = [np.nan] * lead + [float(v) for v in before] + [np.nan] * gap + [float(v) for v in after]
    out = fill_linear_then_rolling_avg(pd.Series(values), linear, rolling)
    return {"name": name, "input": [None if np.isnan(v) else v for v in values], "linear": linear, "rolling": rolling,
            "output": [None if np.isnan(v) else float(v) for v in out.tolist()]}

ramp = lambda a, b: [x * 1.5 + 3 for x in range(a, b)]
cases = [
    case("gap12_10_10", ramp(0, 15), 12, ramp(27, 45), 10, 10),
    case("gap31_10_10", ramp(0, 15), 31, ramp(46, 70), 10, 10),
    case("gap12_2_5", ramp(0, 6), 12, ramp(18, 26), 2, 5),
    case("gap12_2_6", ramp(0, 6), 12, ramp(18, 26), 2, 6),
    case("lead5_2_3", ramp(0, 5), 0, [], 2, 3, lead=5),
    case("gap7_2_3_noisy", [1.0, 4.0, 2.0, 8.0], 7, [3.0, 9.0, 1.0, 6.0, 2.0], 2, 3),
    case("gap12_10_0", ramp(0, 15), 12, ramp(27, 45), 10, 0),
    case("gap25_10_0", ramp(0, 15), 25, ramp(40, 60), 10, 0),
]
json.dump(cases, open("pandas_fill_linear_then_rolling_avg.json", "w"), indent=1)
print("cases:", [c["name"] for c in cases], "pandas", pd.__version__)
