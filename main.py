"""gta6-twitch-ccu-forecast-01 entry, v2 (Python 3.13, standard library only).

Forecasts hourly Twitch concurrent viewers of Grand Theft Auto VI for hours
2..24 of launch day, from the hour-1 print only.

Model (fitted offline on 37 historical big-game launch days from the kit's
comparables; parameters below are hardcoded constants, no data is read at
runtime beyond the case file). Ensemble in log space, weights 0.4 / 0.3 / 0.3:

  Component A (weight 0.4): the v1 model -- per-hour piecewise log-log
    regression (knots at 20k / 200k hour-1 viewers) with tier-scaled
    residual quantiles, averaged 70/30 with a tiered median log-ratio
    model, plus median/q95 caps (2500x / 20000x hour 1).

  Component B (weight 0.3): kernel-weighted local-linear log-ratio
    forecaster. For each hour, a Gaussian-kernel (bandwidth 1.5 in
    ln hour-1) weighted local-linear fit of ln(v_h) on ln(v1) over the
    37 launches; quantiles are a 50/50 shrink of the weighted empirical
    residual quantiles toward Gaussian (mean + z*sd) quantiles, which
    stabilises the tails with only ~37 points. Falls back to the
    global OLS fit when the effective sample is tiny (far outside the
    fitted hour-1 range).

  Component C (weight 0.3): 2-D local-linear forecaster pooling hours.
    One weighted fit per target hour over all 37 launches x 23 hours,
    with Gaussian kernels in ln hour-1 (bandwidth 2.0) and in hour
    (bandwidth 3.0); quantiles from weighted residual quantiles. Pooling
    across neighbouring hours smooths the median curve and the spread.

  The three components' quantiles are averaged in log space, then the
  spread is widened x1.05 around the median (the v1 ensemble was
  slightly under-covered in backtest: 0.844 vs the 0.90 nominal).

Leave-one-launch-out backtest over the same 37 launches (exact server
pinball loss on the log scale, mean over hours x quantiles):
  v1 model            0.3785 (BIG h1>100k 0.2231, MID 0.3226, SMALL 0.7294)
  kernel local-linear 0.3739 (BIG 0.2082, MID 0.3239, SMALL 0.7353)
  hour-pooled 2-D     0.3764 (BIG 0.2164, MID 0.3257, SMALL 0.7282)
  THIS v2 ensemble    0.3686 (BIG 0.2126, MID 0.3144, SMALL 0.7183,
                             typical miss 73.3%, 90% coverage 0.895)
  House persistence baseline on the same 37 launches: 0.9297.

CANDIDATE STATUS: the v2 gate (LOLO < 0.30, material improvement over
v1) was NOT met -- 0.3686 is only 2.6% better than v1, well inside
backtest noise (per-launch losses range 0.1-1.6, dominated by a few
idiosyncratic launches: Wukong 1.50, Marathon 1.29). No subset regresses
(BIG -4.7%, MID -2.5%, SMALL -1.5%), so this file is kept as a ready
candidate for the coordinator to overrule or reject; it has NOT been
submitted and must not be treated as a validated improvement. The 0.30
target looks unreachable from hour-1-only features: shape features
(pre-launch baseline, channels, viewers/channel), k-NN analogs,
unlock-type conditioning and tier-filtered training were all tested and
did not help (see VALIDATION.md).

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


# --- embedded training table for components B and C ---
# ln(hour-1 viewers) and ln(viewers) hours 1..24 for the 37 comparable
# launches (same order in both lists; names in model_params_v2.json).
# These are the kit's comparables -- training data, not practice answers.
LOG_H1 = [11.7582, 7.2513, 11.8981, 8.1453, 7.812, 8.5053, 11.8024, 11.8429, 11.2798, 10.988, 11.6476, 9.2723, 13.6553, 10.8268, 10.6507, 13.1871, 13.6822, 12.6758, 10.8879, 10.6896, 10.1778, 11.7899, 10.2112, 6.0379, 9.9294, 12.8671, 5.0999, 6.0259, 7.2392, 12.0194, 11.9337, 10.7439, 8.7641, 11.9629, 10.2057, 12.8077, 11.7749]
LOG_SERIES = [
[11.7582, 11.5456, 11.2033, 11.1758, 11.1046, 10.9887, 10.7161, 10.7839, 10.8089, 10.7664, 10.8306, 10.8716, 11.1393, 11.3308, 11.4957, 11.574, 11.6749, 11.817, 12.0297, 12.1866, 12.1346, 11.9953, 11.8073, 11.9133],
[7.2513, 6.9838, 7.3278, 7.495, 7.5202, 7.4576, 8.0392, 8.8416, 9.7457, 11.4547, 12.2396, 12.4079, 12.5116, 12.5904, 12.6075, 12.665, 12.7566, 12.7704, 12.8111, 12.7916, 12.7032, 12.7091, 12.6816, 12.6099],
[11.8981, 11.9186, 11.5156, 10.9788, 10.8979, 10.8591, 10.7154, 10.8244, 10.9349, 11.1415, 11.2335, 11.3392, 11.4776, 11.5985, 11.596, 11.596, 11.9211, 11.8351, 11.8683, 11.8861, 11.7484, 11.7405, 11.9433, 11.7746],
[8.1453, 7.874, 7.9073, 7.6094, 7.1253, 6.7719, 7.2449, 7.6396, 8.0839, 8.2641, 8.4, 8.7927, 10.5252, 11.1304, 11.8269, 12.6076, 12.9457, 13.0453, 13.0395, 13.0268, 12.9561, 12.7832, 12.6761, 12.5031],
[7.812, 7.7895, 7.7428, 7.6994, 7.4821, 7.3245, 7.5919, 8.2459, 8.6357, 8.9224, 9.2272, 9.4958, 10.129, 10.8989, 11.8452, 13.0823, 13.2084, 13.2305, 13.2421, 13.1388, 13.0032, 12.8642, 12.6941, 12.5485],
[8.5053, 9.6463, 11.7742, 12.0982, 12.0996, 12.0608, 12.0909, 12.0973, 12.1098, 12.1091, 12.149, 12.3029, 12.5111, 12.6199, 12.738, 12.7144, 12.8416, 12.9014, 12.349, 12.1631, 12.9014, 13.0276, 12.8697, 12.6881],
[11.8024, 12.3202, 11.8749, 11.4484, 11.4106, 11.2806, 11.2611, 11.2079, 11.0722, 10.9312, 11.1033, 11.1507, 11.1239, 11.2329, 11.3644, 11.4311, 11.5214, 11.5946, 11.7887, 11.8348, 11.8508, 11.7303, 11.8001, 11.9545],
[11.8429, 11.7488, 11.6733, 11.6741, 11.9796, 12.0969, 12.0792, 12.0714, 12.0209, 11.9659, 11.9561, 11.9373, 12.0255, 12.1155, 12.2605, 12.4418, 12.5602, 12.6694, 12.7989, 12.8269, 12.7696, 12.8507, 12.8111, 12.7872],
[11.2798, 11.1377, 11.0459, 11.022, 11.0071, 11.1009, 11.1689, 11.2052, 11.1912, 11.2166, 11.2412, 11.247, 11.2862, 11.3173, 11.4321, 11.5849, 11.7456, 11.7276, 11.8586, 11.8586, 11.8111, 11.9068, 11.9362, 11.751],
[10.988, 10.9571, 10.8964, 10.8273, 10.8469, 10.9409, 10.9667, 10.969, 10.9864, 10.9397, 10.9135, 10.9658, 11.1973, 11.3862, 11.4967, 11.6107, 11.6565, 11.7133, 11.8642, 11.8516, 11.8606, 11.8702, 11.8521, 11.8042],
[11.6476, 11.54, 11.4736, 11.36, 11.356, 11.6369, 11.7588, 11.7939, 11.7309, 11.7241, 11.7294, 11.7225, 11.6933, 11.7536, 11.878, 11.8488, 11.8896, 11.9479, 12.0032, 12.0279, 12.1237, 12.1872, 12.1693, 12.1092],
[9.2723, 8.9975, 8.5174, 8.6307, 8.6169, 8.3376, 8.8598, 9.9473, 10.2122, 10.4493, 10.4633, 10.6282, 10.9453, 11.2093, 11.3718, 11.3458, 11.4058, 11.4347, 11.5918, 11.6827, 11.6137, 11.5653, 11.3884, 11.2175],
[13.6553, 13.8825, 13.8111, 13.653, 13.4582, 13.3081, 13.1184, 12.9907, 12.8582, 12.8154, 12.8362, 12.9068, 13.0977, 13.2433, 13.3743, 13.5733, 13.6127, 13.6364, 13.5914, 13.5497, 13.5956, 13.5179, 13.3701, 13.0514],
[10.8268, 10.8042, 10.472, 10.1222, 10.0643, 10.0845, 10.1297, 10.3135, 10.3047, 10.1497, 10.1237, 10.2815, 10.4665, 10.7358, 10.9794, 11.0703, 11.1497, 11.1556, 11.3201, 11.1316, 10.9901, 10.8782, 10.7859, 10.8675],
[10.6507, 10.4127, 10.2945, 10.3589, 10.4081, 10.413, 10.4191, 10.4356, 10.3372, 10.289, 10.221, 10.3855, 10.5797, 10.7781, 10.9834, 11.1136, 11.0862, 11.1875, 11.1393, 11.0852, 11.0522, 10.9851, 10.6899, 10.6429],
[13.1871, 13.1824, 13.1301, 13.0666, 13.0008, 12.9245, 12.915, 12.984, 13.0368, 13.0946, 13.1274, 13.1744, 13.1802, 13.2296, 13.2463, 13.2942, 13.3718, 13.4207, 13.4109, 13.3993, 13.3709, 13.3238, 13.2401, 13.1461],
[13.6822, 13.5942, 13.4627, 13.3075, 13.2612, 13.2252, 13.1813, 13.0884, 13.0436, 13.0482, 13.0926, 13.117, 13.2013, 13.3124, 13.4037, 13.4915, 13.5651, 13.5925, 13.6329, 13.5713, 13.5756, 13.5278, 13.4319, 13.3566],
[12.6758, 12.5485, 12.3085, 12.4035, 12.2354, 11.9996, 11.9474, 11.9684, 11.9438, 11.9381, 11.9829, 11.9572, 12.0444, 12.1526, 12.2679, 12.4345, 12.5614, 12.6546, 12.6725, 12.6218, 12.5975, 12.5788, 12.5429, 12.4767],
[10.8879, 10.9484, 10.7898, 10.6948, 10.709, 10.9964, 10.9328, 10.8284, 10.7017, 10.6453, 10.7233, 10.8064, 10.9454, 11.054, 11.1701, 11.2217, 11.1781, 11.209, 11.2923, 11.5933, 11.6939, 11.6555, 11.4847, 11.158],
[10.6896, 10.3271, 9.9712, 9.8683, 9.4803, 9.5332, 9.8835, 9.98, 9.9486, 10.2342, 10.4454, 10.6138, 10.5695, 10.3825, 10.4373, 10.7875, 10.8952, 11.0005, 10.7865, 10.9355, 10.9676, 11.1581, 11.3521, 10.8622],
[10.1778, 9.892, 9.9008, 9.8733, 10.0546, 10.1567, 10.1355, 10.1278, 10.1762, 10.2799, 10.3987, 10.5518, 10.7045, 10.8965, 10.9738, 11.1252, 11.0612, 11.1102, 11.3757, 11.743, 11.8446, 11.6798, 11.6285, 11.0709],
[11.7899, 11.5968, 11.465, 11.3561, 11.7307, 11.8568, 11.7334, 11.6832, 11.664, 11.6257, 11.6669, 11.7008, 11.9172, 12.1152, 12.2302, 12.2921, 12.3612, 12.458, 12.5072, 12.5986, 12.6023, 12.5671, 12.5309, 12.5021],
[10.2112, 9.8445, 9.0181, 9.3387, 9.9921, 10.1215, 10.2505, 10.3567, 10.4927, 10.4939, 10.5004, 10.5945, 10.8876, 11.0555, 11.19, 11.2109, 11.2283, 11.4917, 11.4722, 11.45, 11.4963, 11.4649, 11.3807, 11.4485],
[6.0379, 5.5452, 4.3307, 4.4773, 4.5747, 5.0814, 5.7494, 6.2066, 8.1215, 10.1308, 10.5348, 10.6079, 10.7706, 10.9993, 11.1855, 11.2375, 11.1498, 11.4546, 11.5273, 11.8137, 11.8867, 11.8614, 11.752, 11.6962],
[9.9294, 10.135, 10.1361, 10.1769, 10.1502, 10.1074, 10.0632, 9.6483, 9.3966, 9.4143, 9.1842, 8.9842, 8.6863, 8.7268, 8.6781, 8.8828, 9.2649, 10.3744, 11.6635, 12.089, 12.0944, 11.8738, 11.7421, 11.495],
[12.8671, 12.7471, 12.3898, 12.2358, 12.1876, 12.1149, 12.0297, 12.115, 12.0327, 12.105, 12.1305, 12.2007, 12.3729, 12.6854, 12.7665, 12.7365, 13.1124, 13.1323, 13.0976, 13.0143, 13.0513, 13.0678, 13.0803, 12.9747],
[5.0999, 4.4998, 4.5109, 4.6151, 4.7958, 4.92, 5.1985, 5.323, 6.1862, 7.9617, 8.2377, 8.5936, 9.6275, 11.3408, 12.7126, 12.7593, 12.7267, 12.7232, 12.6358, 12.5573, 12.483, 12.4001, 12.2896, 12.0919],
[6.0259, 6.3208, 6.5862, 6.7912, 6.9613, 7.069, 7.1499, 7.3902, 8.1167, 8.7129, 8.7749, 9.3352, 9.6476, 9.8283, 10.1285, 11.4193, 12.5573, 12.8284, 12.8761, 12.9213, 13.0332, 12.9065, 12.6725, 12.4929],
[7.2392, 6.9819, 6.2804, 6.4184, 6.0064, 6.0186, 6.0064, 6.073, 6.1738, 6.2558, 6.3026, 6.4184, 6.7639, 7.1341, 7.7723, 9.1971, 9.805, 10.6671, 11.9523, 12.0381, 11.9783, 11.9363, 11.7621, 11.6814],
[12.0194, 12.2913, 12.3559, 12.2552, 12.0955, 12.0006, 11.8318, 11.7037, 11.5916, 11.438, 11.2731, 11.2858, 11.3273, 11.4634, 11.5576, 11.6499, 11.7397, 11.8235, 11.8512, 11.8373, 12.3938, 12.4952, 12.139, 11.8476],
[11.9337, 11.7391, 11.5762, 11.5509, 11.4631, 12.5022, 12.7406, 12.728, 12.7193, 12.7276, 12.6949, 12.6955, 12.77, 12.8555, 12.901, 12.9056, 12.8772, 12.8564, 12.868, 12.8136, 12.7797, 12.7807, 12.7362, 12.6685],
[10.7439, 10.6014, 10.4069, 10.6381, 10.9414, 11.9942, 12.263, 12.2714, 12.1923, 12.1836, 12.2322, 12.3155, 12.4539, 12.5839, 12.698, 12.8912, 12.9564, 13.0498, 13.1926, 13.2407, 13.1844, 13.0516, 12.9575, 12.8512],
[8.7641, 8.6264, 8.4812, 8.454, 8.2777, 8.4884, 8.9016, 9.0132, 9.0332, 8.9001, 9.2573, 9.2909, 9.717, 9.9166, 10.2419, 10.384, 10.3335, 10.4549, 10.4923, 10.4091, 10.377, 10.3648, 10.0466, 9.8071],
[11.9629, 12.0219, 12.0117, 11.8778, 11.4799, 11.402, 11.2921, 11.3799, 11.2621, 11.1073, 10.9345, 10.9589, 11.4375, 11.415, 11.4208, 11.3698, 11.4116, 11.4908, 11.5925, 11.6129, 11.53, 11.5413, 11.6391, 11.614],
[10.2057, 9.9602, 9.4647, 9.55, 10.1314, 10.2008, 9.7457, 9.6643, 9.9082, 10.0082, 10.0592, 10.2335, 10.315, 10.469, 10.9599, 11.2073, 11.1586, 11.1756, 11.4294, 11.3832, 11.3642, 11.3344, 11.2356, 11.1092],
[12.8077, 12.5041, 12.1841, 12.0209, 12.4647, 12.4027, 12.2964, 12.2081, 12.1251, 12.0872, 12.1233, 12.2746, 12.4723, 12.6253, 12.7864, 12.8775, 12.988, 13.0234, 13.0775, 13.0987, 13.0524, 12.9461, 12.962, 12.9635],
[11.7749, 11.592, 11.3693, 11.1055, 11.849, 11.9269, 11.9168, 11.8332, 11.8005, 11.9205, 12.0236, 12.1706, 12.2686, 12.4238, 12.5818, 12.5913, 12.7045, 12.7311, 12.755, 12.7469, 12.6846, 12.6071, 12.5552, 12.418],
]

KLL_BW = 1.5
POOL_BW_X = 2.0
POOL_BW_H = 3.0
W_V1 = 0.4
W_KLL = 0.3
W_POOL = 0.3
SPREAD = 1.05
GAUSS_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def _wls(X, y, w):
    """Weighted least squares (Gaussian elimination, tiny ridge)."""
    k = len(X[0])
    A = [[sum(w[r] * X[r][i] * X[r][j] for r in range(len(X))) for j in range(k)] for i in range(k)]
    b = [sum(w[r] * X[r][i] * y[r] for r in range(len(X))) for i in range(k)]
    for i in range(k):
        A[i][i] += 1e-8
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    n = k
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        if abs(pv) < 1e-12:
            continue
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col] / pv
                for cc in range(col, n + 1):
                    M[r][cc] -= f * M[col][cc]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def _emp_quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    f = pos - lo
    return sorted_vals[lo] * (1 - f) + sorted_vals[hi] * f


def _wquantile(pairs, tau):
    """Weighted empirical quantile of (value, weight) pairs."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * tot
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc >= target:
            if i == 0:
                return v
            prev_v, _prev_w = pairs[i - 1]
            frac = (target - (acc - w)) / w if w > 0 else 0.0
            return prev_v + frac * (v - prev_v)
    return pairs[-1][0]


def _eff_n(ws):
    s = sum(ws)
    s2 = sum(w * w for w in ws)
    return (s * s / s2) if s2 > 0 else 0.0


def _gauss(u):
    return math.exp(-0.5 * u * u)


def kll_log_quantiles(p1, h):
    """Component B: kernel local-linear log-quantiles for one hour."""
    x0 = math.log(p1)
    idx = h - 1
    n = len(LOG_H1)
    ws = [_gauss((LOG_H1[i] - x0) / KLL_BW) for i in range(n)]
    X = [[1.0, LOG_H1[i] - x0] for i in range(n)]
    y = [LOG_SERIES[i][idx] for i in range(n)]
    en = _eff_n(ws)
    if en < 8:
        ws = [_gauss((LOG_H1[i] - x0) / (KLL_BW * 1.8)) for i in range(n)]
        en = _eff_n(ws)
    if en < 5:
        beta = _wls([[1.0, LOG_H1[i]] for i in range(n)], y, [1.0] * n)
        mu = beta[0] + beta[1] * x0
        resid = sorted(y[i] - (beta[0] + beta[1] * LOG_H1[i]) for i in range(n))
        rq = [_emp_quantile(resid, t) for t in LEVELS]
        return [mu + r for r in rq]
    beta = _wls(X, y, ws)
    mu = beta[0]
    resid = [(y[i] - (beta[0] * X[i][0] + beta[1] * X[i][1]), ws[i]) for i in range(n)]
    emp = [_wquantile(resid, t) for t in LEVELS]
    tot = sum(w for _, w in resid)
    m = sum(w * r for r, w in resid) / tot
    sd = math.sqrt(sum(w * (r - m) ** 2 for r, w in resid) / tot) or 0.5
    gau = [m + z * sd for z in GAUSS_Z]
    rq = [0.5 * a + 0.5 * b for a, b in zip(emp, gau)]
    rq[2] = 0.5 * emp[2] + 0.5 * m
    q = [mu + r for r in rq]
    for i in range(1, 5):
        q[i] = max(q[i], q[i - 1])
    return q


def pool_log_quantiles(p1, h0):
    """Component C: hour-pooled 2-D local-linear log-quantiles."""
    x0 = math.log(p1)
    n = len(LOG_H1)
    ws, X, y = [], [], []
    for i in range(n):
        wx = _gauss((LOG_H1[i] - x0) / POOL_BW_X)
        for h in range(2, 25):
            ws.append(wx * _gauss((h - h0) / POOL_BW_H))
            X.append([1.0, LOG_H1[i] - x0, h - h0])
            y.append(LOG_SERIES[i][h - 1])
    beta = _wls(X, y, ws)
    mu = beta[0]
    resid = [(y[i] - (beta[0] * X[i][0] + beta[1] * X[i][1] + beta[2] * X[i][2]), ws[i])
             for i in range(len(X))]
    rq = [_wquantile(resid, t) for t in LEVELS]
    q = [mu + r for r in rq]
    for i in range(1, 5):
        q[i] = max(q[i], q[i - 1])
    return q


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


def v1_hour_quantiles(p1: float, h: int) -> list:
    """Component A: the v1 model (piecewise log-log + tiered median-ratio)."""
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


def hour_quantiles(p1: float, h: int) -> list:
    """v2 ensemble (log space) for one hour, as five viewer counts:
    0.4 * v1 model + 0.3 * kernel local-linear + 0.3 * hour-pooled
    local-linear, then the spread widened x1.05 around the median."""
    try:
        lq_v1 = [math.log(max(v, 1e-9)) for v in v1_hour_quantiles(p1, h)]
    except Exception:
        lq_v1 = None
    lq_kll = kll_log_quantiles(p1, h)
    lq_pool = pool_log_quantiles(p1, h)
    if lq_v1 is None:
        lq_v1 = lq_kll
    raw = []
    for i in range(5):
        v = math.exp(W_V1 * lq_v1[i] + W_KLL * lq_kll[i] + W_POOL * lq_pool[i])
        if not math.isfinite(v) or v <= 0:
            v = p1
        raw.append(min(v, 1e9))
    med = math.log(max(raw[2], 1e-9))
    out = []
    prev = 0.0
    for v in raw:
        lv = med + SPREAD * (math.log(max(v, 1e-9)) - med)
        v = math.exp(lv)
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
