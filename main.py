"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts hourly Twitch concurrent viewers of Grand Theft Auto VI for hours
2..24 of launch day, from the hour-1 print only.

Model (fitted offline on 37 historical big-game launch days from the kit's
comparables; parameters below are hardcoded constants, no data is read at
runtime beyond the case file):

  Component A (weight 0.7): per-hour piecewise log-log regression.
    ln(v_h) ~ b0 + b1*ln(v1) + b2*max(0, ln(v1)-ln(20000))
              + b3*max(0, ln(v1)-ln(200000)) + residual-quantile.
    The hinges capture the key empirical fact that a small hour-1 print
    (unlocks still pending in most regions) precedes explosive growth
    (median peak ~200x hour 1), while a large hour-1 print (already
    unlocked in NZ/CET/UK) precedes a near-flat day (median peak ~1.2x).
    Residual quantiles per hour are scaled by h1-tier (small/mid/large):
    small-h1 launches are far more volatile, so their intervals are wider.

  Component B (weight 0.3): tiered median log-ratio model.
    ln(v_h/v1) median per h1-tier (<10k / 10k-100k / >100k) plus the
    empirical residual quantiles of the log-ratio around that median.

  Ensemble: the five quantiles of A and B are averaged in log space with
  weights 0.7 / 0.3, then exponentiated. Quantiles are clamped to
  (0, 1e9] and forced non-decreasing.

Leave-one-launch-out backtest over the same 37 launches (exact server
pinball loss on the log scale, mean over hours x quantiles):
  persistence (house baseline) 0.930 | global median-ratio 0.714
  log-log regression 0.438 | tiered median-ratio 0.468
  piecewise log-log 0.403 | piecewise + tier-spread 0.382
  FINAL ensemble (this file) 0.379 -- best of all candidates tested,
  and best in the large-h1 regime too (0.223 vs persistence 0.93+).

The program is deterministic (no randomness, no network, no time use),
reads $MAKEFUN_INPUT_DIR (default /input) and writes $MAKEFUN_OUTPUT_DIR
(default /output), one <id>.out per <id>.in. Any malformed case still gets
a valid persistence-style forecast instead of a crash.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- fitted constants (37 launches, leave-one-out winner) ---
KNOT1 = math.log(20000.0)
KNOT2 = math.log(200000.0)
TIER_EDGES = (10000.0, 100000.0)
W_A = 0.7  # component A weight; component B gets 0.3

# A: per-hour OLS coefficients [b0, b1, b2, b3], hours 2..24.
BETA = {
    2: [-0.4156, 1.0360, -0.0387, 0.0418],
    3: [-0.5313, 1.0406, -0.1204, 0.1996],
    4: [-0.3569, 1.0299, -0.2227, 0.3765],
    5: [-0.5698, 1.0576, -0.2521, 0.2848],
    6: [-0.4227, 1.0472, -0.1960, 0.0191],
    7: [0.3114, 0.9860, -0.2220, 0.0932],
    8: [1.0891, 0.9213, -0.2318, 0.1592],
    9: [3.5760, 0.6653, 0.0322, 0.1271],
    10: [6.8719, 0.3219, 0.4286, 0.0628],
    11: [7.6066, 0.2529, 0.4766, 0.1109],
    12: [8.2311, 0.1993, 0.5077, 0.1451],
    13: [9.5116, 0.0892, 0.5855, 0.1822],
    14: [11.1331, -0.0662, 0.7615, 0.1672],
    15: [12.5686, -0.1950, 0.8615, 0.1993],
    16: [13.7468, -0.2893, 0.8537, 0.3862],
    17: [14.4902, -0.3589, 0.9488, 0.3575],
    18: [14.6076, -0.3465, 0.8288, 0.5039],
    19: [14.4331, -0.3031, 0.6819, 0.5920],
    20: [14.3105, -0.2817, 0.6249, 0.5844],
    21: [14.3043, -0.2778, 0.6088, 0.5969],
    22: [14.2566, -0.2799, 0.6448, 0.5085],
    23: [14.0882, -0.2766, 0.7002, 0.3806],
    24: [14.1235, -0.3027, 0.8115, 0.1976],
}

# A: per-hour empirical residual quantiles (log units), hours 2..24.
A_RQ = {
    2: [-0.3018, -0.1485, -0.0798, 0.0855, 0.5116],
    3: [-0.7848, -0.2431, -0.0828, 0.1389, 0.6774],
    4: [-0.6951, -0.3931, -0.0098, 0.1457, 0.7516],
    5: [-1.0621, -0.2927, -0.0283, 0.1716, 0.6417],
    6: [-1.0949, -0.3682, -0.0185, 0.2106, 1.2115],
    7: [-0.9269, -0.4219, -0.0541, 0.2165, 1.2000],
    8: [-0.8113, -0.4455, -0.0691, 0.3152, 1.1872],
    9: [-0.8114, -0.4172, -0.1377, 0.3494, 1.3645],
    10: [-0.8801, -0.4640, -0.1531, 0.3555, 1.6443],
    11: [-1.0100, -0.3930, -0.1732, 0.3459, 1.6847],
    12: [-1.0972, -0.3967, -0.0968, 0.2845, 1.6885],
    13: [-0.8061, -0.3467, -0.0793, 0.2872, 1.6419],
    14: [-1.0787, -0.3841, 0.0273, 0.3256, 1.6047],
    15: [-1.4075, -0.3711, 0.0219, 0.3473, 1.4622],
    16: [-1.0642, -0.4109, -0.0167, 0.3463, 1.4500],
    17: [-1.2757, -0.3379, -0.0016, 0.2439, 1.4280],
    18: [-1.0716, -0.3725, -0.0947, 0.3090, 1.2743],
    19: [-0.9693, -0.3187, -0.0889, 0.2694, 1.0955],
    20: [-0.8078, -0.3166, -0.0915, 0.3082, 1.0144],
    21: [-0.8509, -0.3153, -0.0521, 0.4129, 0.9232],
    22: [-0.8065, -0.2942, -0.0728, 0.3796, 0.8753],
    23: [-0.9584, -0.3237, -0.0795, 0.3822, 0.8995],
    24: [-0.7549, -0.3546, -0.0749, 0.3402, 0.9040],
}

# A: per-hour residual-spread scale per h1-tier [small, mid, large].
A_SCALE = {
    2: [1.6059, 0.5704, 0.7021],
    3: [1.8026, 0.5211, 0.3659],
    4: [1.8323, 0.4560, 0.3926],
    5: [1.8525, 0.4927, 0.3792],
    6: [1.7179, 0.7195, 0.4909],
    7: [1.6560, 0.7169, 0.5962],
    8: [1.6613, 0.7306, 0.5688],
    9: [1.6890, 0.7147, 0.5552],
    10: [1.7769, 0.6100, 0.5073],
    11: [1.7929, 0.6051, 0.4785],
    12: [1.7774, 0.6405, 0.4767],
    13: [1.7156, 0.7708, 0.4498],
    14: [1.6933, 0.8096, 0.4517],
    15: [1.6687, 0.8436, 0.4561],
    16: [1.5664, 0.9291, 0.5137],
    17: [1.5863, 0.8891, 0.5248],
    18: [1.5758, 0.8294, 0.5976],
    19: [1.4281, 0.9698, 0.6481],
    20: [1.3631, 1.0448, 0.6512],
    21: [1.3977, 1.0292, 0.6121],
    22: [1.4190, 0.9717, 0.6518],
    23: [1.4672, 0.9840, 0.5847],
    24: [1.4767, 0.9671, 0.5860],
}

# B: per-hour median log-ratio ln(v_h/v1) per h1-tier [small, mid, large].
B_MED = {
    2: [-0.2573, -0.1902, -0.1009],
    3: [-0.2380, -0.3458, -0.2721],
    4: [-0.3100, -0.2981, -0.3643],
    5: [-0.3299, -0.1990, -0.3674],
    6: [-0.2757, -0.0684, -0.3761],
    7: [0.0986, -0.0766, -0.5060],
    8: [0.2492, -0.0671, -0.5884],
    9: [1.0863, -0.1374, -0.6606],
    10: [2.6871, -0.1303, -0.6772],
    11: [2.7491, -0.1105, -0.6745],
    12: [3.3093, -0.0276, -0.5620],
    13: [3.6218, 0.0834, -0.4507],
    14: [3.8024, 0.2147, -0.2410],
    15: [4.1027, 0.4207, -0.1816],
    16: [5.1997, 0.5428, -0.1063],
    17: [5.1119, 0.5672, 0.1016],
    18: [5.4168, 0.6310, 0.1372],
    19: [5.4302, 1.0370, 0.2271],
    20: [5.3269, 1.0205, 0.1797],
    21: [5.1913, 1.0155, 0.2145],
    22: [5.0523, 1.0054, 0.1696],
    23: [4.8821, 0.9470, 0.0863],
    24: [4.7365, 0.8546, 0.1298],
}

# B: per-hour empirical residual quantiles of the log-ratio (log units).
B_RQ = {
    2: [-0.2093, -0.0845, 0.0, 0.1598, 0.5655],
    3: [-0.7461, -0.2052, 0.0, 0.2477, 0.6466],
    4: [-0.5589, -0.3051, 0.0, 0.1955, 0.6952],
    5: [-0.9245, -0.1565, 0.0, 0.1810, 0.7321],
    6: [-0.9735, -0.3001, 0.0, 0.3574, 1.3188],
    7: [-0.7834, -0.3313, 0.0, 0.3185, 1.3695],
    8: [-0.6650, -0.1480, 0.0, 0.4817, 1.4251],
    9: [-0.8833, -0.1760, 0.0, 0.6861, 1.4741],
    10: [-2.5544, -0.1939, 0.0, 0.5847, 1.4801],
    11: [-2.3035, -0.1446, 0.0, 0.6148, 1.6286],
    12: [-2.6860, -0.1865, 0.0, 0.4882, 1.4266],
    13: [-1.5950, -0.1807, 0.0, 0.5780, 1.5970],
    14: [-1.6638, -0.3057, 0.0, 0.5136, 1.6447],
    15: [-1.8626, -0.2564, 0.0, 0.5581, 1.5625],
    16: [-1.9199, -0.2631, 0.0, 0.4047, 1.5455],
    17: [-1.4946, -0.2770, 0.0, 0.3934, 1.5822],
    18: [-1.2143, -0.3022, 0.0, 0.3389, 1.5601],
    19: [-1.0693, -0.4582, 0.0, 0.2240, 1.4134],
    20: [-0.9535, -0.3617, 0.0, 0.5447, 1.4947],
    21: [-0.8066, -0.3642, 0.0, 0.6316, 1.5032],
    22: [-0.7277, -0.3272, 0.0, 0.6076, 1.4075],
    23: [-0.9238, -0.3366, 0.0, 0.6547, 1.3662],
    24: [-0.8236, -0.3834, 0.0, 0.5824, 1.3483],
}


def tier_of(p1: float) -> int:
    if p1 < TIER_EDGES[0]:
        return 0
    if p1 < TIER_EDGES[1]:
        return 1
    return 2


def round2(x: float) -> float | int:
    """Round to 2 decimals, half up (same convention as the starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


# Guard against log-log extrapolation far below the fitted range
# (smallest fitted hour-1 print was 164; max observed peak/hour-1 ratio
# was ~2120x): cap the median at 2500x hour 1 and q95 at 20000x hour 1.
# Backtest cost is zero (LOLO loss unchanged at 0.3785), it only trims
# absurd outputs if hour 1 ever came in near zero.
MED_CAP = 2500.0
Q95_CAP = 20000.0


def hour_quantiles(p1: float, h: int) -> list:
    """Ensemble forecast (log space) for one hour, as five viewer counts."""
    x = math.log(p1)
    hinge1 = max(0.0, x - KNOT1)
    hinge2 = max(0.0, x - KNOT2)
    tier = tier_of(p1)
    beta = BETA[h]
    mu = beta[0] + beta[1] * x + beta[2] * hinge1 + beta[3] * hinge2
    rq_a = A_RQ[h]
    scale = A_SCALE[h][tier]
    med_a = mu + rq_a[2]
    log_a = [med_a + scale * (r - rq_a[2]) for r in rq_a]
    rq_b = B_RQ[h]
    med_b = x + B_MED[h][tier]  # B_RQ median is 0 by construction
    log_b = [med_b + r for r in rq_b]
    raw = []
    for i in range(5):
        v = math.exp(W_A * log_a[i] + (1.0 - W_A) * log_b[i])
        if not math.isfinite(v) or v <= 0:
            v = p1
        raw.append(min(v, 1e9))
    # Caps: quantiles at/below the median share the median cap, the upper
    # two share the q95 cap; then enforce non-decreasing order.
    for i in range(3):
        raw[i] = min(raw[i], MED_CAP * p1)
    for i in range(3, 5):
        raw[i] = min(raw[i], Q95_CAP * p1)
    out = []
    prev = 0.0
    for v in raw:
        v = max(v, prev)
        prev = v
        out.append(round2(v))
    return out


def fallback_quantiles(p1: float, h: int) -> list:
    """Persistence with widening log-normal spread (house-baseline shape)."""
    z = [-1.645, -0.674, 0.0, 0.674, 1.645]
    sig = 0.15 * math.sqrt(h - 1)
    out, prev = [], 0.0
    for zi in z:
        v = min(max(p1 * math.exp(zi * sig), 1.0), 1e9)
        v = max(v, prev)
        prev = v
        out.append(round2(v))
    return out


def hour1_print(case: dict) -> float:
    try:
        for row in case.get("known") or []:
            if row.get("hour") == 1:
                v = float(row.get("viewers") or 0)
                if math.isfinite(v) and v > 0:
                    return v
    except (TypeError, ValueError):
        pass
    return 1.0


def forecast_case(case: dict) -> dict:
    p1 = hour1_print(case)
    hours = []
    try:
        rows = case.get("forecast") or []
        want = sorted(int(r["hour"]) for r in rows)
    except (TypeError, ValueError, KeyError):
        want = list(range(2, 25))
    if want != list(range(2, 25)):
        want = list(range(2, 25))
    for h in want:
        try:
            q = hour_quantiles(p1, h)
        except Exception:
            q = fallback_quantiles(p1, h)
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            forecast = {"hours": [{"hour": h, "q": fallback_quantiles(1.0, h)} for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
