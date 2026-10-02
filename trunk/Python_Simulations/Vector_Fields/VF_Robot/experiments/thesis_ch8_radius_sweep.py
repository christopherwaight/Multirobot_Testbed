"""
thesis_ch8_radius_sweep.py

Terminal estimation error against formation radius, first order against
second order, for thesis Chapter 8 (Section "Noise Growth with Fit Order").
Estimator-level Monte Carlo on the steady double gyre (A = 0.1); no
closed-loop control.

At each radius rho the formation is centered on a feature and given a
random heading. For each trial the fit is run twice with the same heading,
once noise-free and once with measurement noise sigma on every reading, so

    bias    = RMS error of the noise-free fit (truncation) over headings
    scatter = std of (noisy error - noise-free error), the noise alone

Quantities, all converted to position units:
  first order   triangle on a critical point, delta = p_hat* - p*,
                p_hat* = p_c - J_hat^-1 v_hat_c
  second order  pentagon plus center on the separatrix point (0, 0.25),
                D tracker:  delta_n = d_x D_hat / (2 pi^6 A^2)
                s1 tracker: delta_n = d_x s1_hat / (pi^4 A |cos pi y_f|)
                (true transverse curvatures; the true transverse gradients
                vanish on the separatrix)

Prediction: scatter independent of rho at first order, scatter ~ rho^-2 at
second order. First-order check: per-component scatter ~ ||J^-1|| sigma / sqrt(3).

Outputs: experiments/outputs/thesis_ch8/radius_sweep.json, radius_sweep.png

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/thesis_ch8_radius_sweep.py
"""
import os
import sys
import json
import math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.fields.environments.Double_Gyre import double_gyre_static
from src.control.pentagon_primitives import (_fit_vector_quadratic, _det_gradient,
                                             _strain_quantities)

OUT = os.path.join(ROOT, 'experiments', 'outputs', 'thesis_ch8')
A = 0.1
SIGMA = 0.002
N_TRIALS = 1000
RHO_NOM = 0.075
RHO_MULT = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
SEED = 20260930


def field(pts):
    return np.array([double_gyre_static(x, y) for x, y in pts])


def ring(n, r, phase):
    return np.array([[r * math.cos(phase + 2 * math.pi * j / n),
                      r * math.sin(phase + 2 * math.pi * j / n)] for j in range(n)])


def first_order_pstar(rel, uv):
    Amat = np.column_stack([np.ones(len(rel)), rel])
    cu = np.linalg.solve(Amat, uv[:, 0])
    cv = np.linalg.solve(Amat, uv[:, 1])
    J = np.array([[cu[1], cu[2]], [cv[1], cv[2]]])
    vc = np.array([cu[0], cv[0]])
    return -np.linalg.solve(J, vc)          # relative to the centroid


def second_order_dn(rel, uv, y_f):
    tu, tv = _fit_vector_quadratic(rel, uv[:, 0], uv[:, 1])
    dn_D = _det_gradient(tu, tv)[0] / (2 * math.pi**6 * A**2)
    _, g_s1, _, _, _ = _strain_quantities(tu, tv)
    dn_s1 = g_s1[0] / (math.pi**4 * A * abs(math.cos(math.pi * y_f)))
    return dn_D, dn_s1


def run():
    rng = np.random.default_rng(SEED)
    targets = {'first order, gyre center (0.5, 0)': np.array([0.5, 0.0]),
               'first order, saddle (0, 0.5)': np.array([0.0, 0.5])}
    sep_pt = np.array([0.0, 0.25])
    y_f = sep_pt[1] + 0.5
    Jinv_norm = 1.0 / (math.pi**2 * A)        # both first-order targets
    rows = []
    for m in RHO_MULT:
        rho = m * RHO_NOM
        rec = {'rho': rho, 'rho_mult': m}
        # first order
        for name, p in targets.items():
            e0, en = [], []
            for _ in range(N_TRIALS):
                rel = ring(3, rho, rng.uniform(0, 2 * math.pi))
                uv = field(rel + p)
                noise = rng.normal(0, SIGMA, uv.shape)
                e0.append(first_order_pstar(rel, uv))
                en.append(first_order_pstar(rel, uv + noise))
            e0, en = np.array(e0), np.array(en)
            d = en - e0
            rec[name] = {'bias': float(np.sqrt(np.mean(np.sum(e0**2, axis=1)))),
                         'bias_max': float(np.max(np.linalg.norm(e0, axis=1))),
                         'scatter_per_component': float(np.mean(d.std(axis=0))),
                         'predicted_per_component': Jinv_norm * SIGMA / math.sqrt(3)}
        # second order
        eD0, eDn, eS0, eSn = [], [], [], []
        for _ in range(N_TRIALS):
            rel = np.vstack([[0.0, 0.0], ring(5, rho, rng.uniform(0, 2 * math.pi))])
            uv = field(rel + sep_pt)
            noise = rng.normal(0, SIGMA, uv.shape)
            a0, b0 = second_order_dn(rel, uv, y_f)
            a1, b1 = second_order_dn(rel, uv + noise, y_f)
            eD0.append(a0); eDn.append(a1); eS0.append(b0); eSn.append(b1)
        for name, z0, zn in (('second order, D tracker', eD0, eDn),
                             ('second order, s1 tracker', eS0, eSn)):
            z0, zn = np.array(z0), np.array(zn)
            rec[name] = {'bias': float(np.sqrt(np.mean(z0**2))), 'bias_max': float(np.max(np.abs(z0))),
                         'scatter': float((zn - z0).std())}
        rows.append(rec)
        print(f'rho = {rho:.4f} done')
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = run()
    names1 = ['first order, gyre center (0.5, 0)', 'first order, saddle (0, 0.5)']
    names2 = ['second order, D tracker', 'second order, s1 tracker']
    rhos = np.array([r['rho'] for r in rows])
    print(f"\nsigma = {SIGMA}, {N_TRIALS} trials per point\n")
    print(f"{'rho':>7} | " + ' | '.join(f'{n:>34}' for n in names1 + names2))
    print(f"{'':>7} | " + ' | '.join(f"{'bias':>11} {'scatter':>11} {'pred':>10}" for _ in names1)
          + ' | ' + ' | '.join(f"{'bias':>16} {'scatter':>17}" for _ in names2))
    for r in rows:
        s = f"{r['rho']:>7.4f} | "
        s += ' | '.join(f"{r[n]['bias']:>11.2e} {r[n]['scatter_per_component']:>11.2e} "
                        f"{r[n]['predicted_per_component']:>10.2e}" for n in names1)
        s += ' | ' + ' | '.join(f"{r[n]['bias']:>16.2e} {r[n]['scatter']:>17.2e}" for n in names2)
        print(s)
    slopes = {}
    for n in names1:
        slopes[n] = float(np.polyfit(np.log(rhos), np.log([r[n]['scatter_per_component'] for r in rows]), 1)[0])
    for n in names2:
        slopes[n] = float(np.polyfit(np.log(rhos), np.log([r[n]['scatter'] for r in rows]), 1)[0])
    bias_slopes = {n: float(np.polyfit(np.log(rhos), np.log([max(r[n]['bias'], 1e-300) for r in rows]), 1)[0])
                   for n in names1 + names2}
    print('\nlog-log slope of scatter against rho:', {k: round(v, 3) for k, v in slopes.items()})
    print('log-log slope of bias against rho:   ', {k: round(v, 3) for k, v in bias_slopes.items()})
    json.dump({'sigma': SIGMA, 'n_trials': N_TRIALS, 'seed': SEED, 'rows': rows,
               'scatter_slopes': slopes, 'bias_slopes': bias_slopes},
              open(os.path.join(OUT, 'radius_sweep.json'), 'w'), indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2), sharex=True)
    for n, mk in zip(names1, ('o', 's')):
        ax[0].loglog(rhos, [r[n]['scatter_per_component'] for r in rows], mk + '-', label=n.replace('first order, ', ''))
        ax[1].loglog(rhos, [max(r[n]['bias'], 1e-12) for r in rows], mk + '-', label=n.replace('first order, ', ''))
    for n, mk in zip(names2, ('^', 'v')):
        ax[0].loglog(rhos, [r[n]['scatter'] for r in rows], mk + '-', label=n.replace('second order, ', ''))
        ax[1].loglog(rhos, [max(r[n]['bias'], 1e-12) for r in rows], mk + '-', label=n.replace('second order, ', ''))
    ax[0].set_title(f'noise scatter, $\\sigma = {SIGMA}$')
    ax[1].set_title('truncation bias (RMS), $\\sigma = 0$')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    for a in ax:
        a.xaxis.set_major_locator(FixedLocator(rhos))
        a.xaxis.set_major_formatter(FixedFormatter([f'{r:.3f}' for r in rhos]))
        a.xaxis.set_minor_locator(NullLocator())
        a.tick_params(axis='x', labelrotation=45, labelsize=7)
        a.set_xlabel(r'formation radius $\rho$')
        a.set_ylabel('terminal error (length)')
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'radius_sweep.png'), dpi=200)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
