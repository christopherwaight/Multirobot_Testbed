import csv, sys

def load_summary(path, key_cols, val_col):
    out = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#')]
    reader = csv.DictReader(lines)
    for row in reader:
        key = tuple(float(row[c]) for c in key_cols)
        out[key] = float(row[val_col])
    return out

def compare(name, path_1000, path_10000, val_col='success_traverse'):
    a = load_summary(path_1000, ['sigma_uv','sigma_p'], val_col)
    b = load_summary(path_10000, ['sigma_uv','sigma_p'], val_col)
    common = sorted(set(a) & set(b))
    print(f"\n=== {name} ({val_col}) ===")
    print(f"{'sigma_uv':>9} {'sigma_p':>8} {'1000-trial':>11} {'10000-trial':>12} {'delta(pp)':>10}")
    max_delta = 0
    for k in common:
        v1000, v10000 = a[k]*100, b[k]*100
        d = v10000 - v1000
        max_delta = max(max_delta, abs(d))
        flag = "  <-- >3pp" if abs(d) > 3 else ""
        print(f"{k[0]:9.3f} {k[1]:8.3f} {v1000:11.1f} {v10000:12.1f} {d:+10.1f}{flag}")
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    print(f"Max |delta| = {max_delta:.1f} pp. Cells only in 1000-trial: {len(only_a)}. Cells only in 10000-trial (new extension rows): {len(only_b)}")
    return max_delta

compare("SEPARATRIX fixed-start (traverse)",
        "experiments/outputs/mc_separatrix/summary_fixed_1000.csv",
        "experiments/outputs/mc_separatrix/summary_fixed.csv",
        "success_traverse")

compare("SEPARATRIX fixed-start (straddle)",
        "experiments/outputs/mc_separatrix/summary_fixed_1000.csv",
        "experiments/outputs/mc_separatrix/summary_fixed.csv",
        "success_straddle")

compare("SEPARATRIX random-start basin (traverse)",
        "experiments/outputs/mc_separatrix/summary_random_basin_1000.csv",
        "experiments/outputs/mc_separatrix/summary_random.csv",
        "success_traverse")

compare("OW fixed-start (track)",
        "experiments/outputs/mc_ow/summary_fixed_1000.csv",
        "experiments/outputs/mc_ow/summary_fixed.csv",
        "success_track")

compare("OECS fixed-start (core)",
        "experiments/outputs/mc_oecs/summary_fixed_1000.csv",
        "experiments/outputs/mc_oecs/summary_fixed.csv",
        "success_core")
