"""
thesis_ch8_geometry.py

Sampling geometry of the order-k polynomial fit, for thesis Chapter 8.
Linear algebra only; no simulation.

For each formation and fit order k, builds the formation matrix Phi_k with
the basis x^a y^b / (a! b!), a + b <= k (the thesis Section 4.4 basis; for
k = 2 it is Draft 11's basis up to column order). Formations are placed at
radius rho = 1, which is the radius-normalized matrix Draft 11 reports.

Reports, per formation:
  - rank and condition number kappa(Phi_k)
  - when rank-deficient, the null vector written as the polynomial curve
    through the robots
  - coefficient noise gains sqrt(diag((Phi^T Phi)^-1)), grouped by order
  - fault tolerance: whether full rank survives the loss of each robot
And, for k = 2, the noise gain of the second-order coefficients against the
number of robots N for (N-1)-gon plus center formations.

Outputs: experiments/outputs/thesis_ch8/geometry.json, geometry_gain_vs_N.png

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/thesis_ch8_geometry.py
"""
import os
import json
import math
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'thesis_ch8')


def monomials(k):
    return [(a, q - a) for q in range(k + 1) for a in range(q, -1, -1)]


def phi_matrix(pts, k):
    mons = monomials(k)
    return np.array([[x**a * y**b / (math.factorial(a) * math.factorial(b))
                      for a, b in mons] for x, y in pts]), mons


def ring(n, r=1.0, phase=0.0):
    return [(r * math.cos(phase + 2 * math.pi * j / n),
             r * math.sin(phase + 2 * math.pi * j / n)) for j in range(n)]


def centered(pts):
    p = np.array(pts, dtype=float)
    p -= p.mean(axis=0)
    return [tuple(v) for v in p / np.max(np.linalg.norm(p, axis=1))]


def poly_str(vec, mons, tol=1e-6):
    # basis coefficients -> monomial coefficients (undo the 1/(a! b!) scaling)
    vec = np.array([c / (math.factorial(a) * math.factorial(b)) for c, (a, b) in zip(vec, mons)])
    vec = vec / np.max(np.abs(vec))
    terms = []
    for c, (a, b) in zip(vec, mons):
        if abs(c) < tol:
            continue
        m = ''.join(s + (f'^{e}' if e > 1 else '') for s, e in (('x', a), ('y', b)) if e)
        terms.append(f'{c:+.3f}{m}')
    return ' '.join(terms) + ' = 0'


def analyze(name, pts, k):
    Phi, mons = phi_matrix(pts, k)
    n_k = len(mons)
    sv = np.linalg.svd(Phi, compute_uv=False)
    rank = int(np.sum(sv > 1e-9 * sv[0]))
    rec = {'formation': name, 'k': k, 'N': len(pts), 'n_k': n_k, 'rank': rank,
           'full_rank': rank == n_k}
    if rank < n_k:
        rec['kappa'] = float('inf')
        _, _, Vt = np.linalg.svd(Phi)
        rec['null_dim'] = n_k - rank
        rec['null_curve'] = poly_str(Vt[-1], mons)
    else:
        rec['kappa'] = float(sv[0] / sv[-1])
        G = np.sqrt(np.diag(np.linalg.inv(Phi.T @ Phi)))
        rec['gain_by_order'] = {q: float(np.mean([g for g, (a, b) in zip(G, mons) if a + b == q]))
                                for q in range(k + 1)}
        # fault tolerance: remove each robot in turn
        survive = []
        for i in range(len(pts)):
            sub = [p for j, p in enumerate(pts) if j != i]
            if len(sub) < n_k:
                survive.append(False)
                continue
            s = np.linalg.svd(phi_matrix(sub, k)[0], compute_uv=False)
            survive.append(bool(np.sum(s > 1e-9 * s[0]) == n_k))
        rec['survives_any_single_loss'] = all(survive)
        rec['fatal_losses'] = [i + 1 for i, ok in enumerate(survive) if not ok]
    return rec


def two_ring(N, r_in=0.5):
    # center robot, inner ring, outer ring rotated half a step
    n_in = (N - 1) // 2
    n_out = N - 1 - n_in
    return [(0.0, 0.0)] + ring(n_in, r_in) + ring(n_out, 1.0, math.pi / n_out)


def best_ten(k=3):
    # grid search over center + two-ring layouts of 10 robots for min kappa
    best = None
    for c in (0, 1):
        for n_in in range(3, 7):
            n_out = 10 - c - n_in
            if n_out < 4:
                continue
            for r_in in np.arange(0.2, 0.96, 0.05):
                for ph in np.linspace(0, 2 * math.pi / n_out, 12, endpoint=False):
                    pts = [(0.0, 0.0)] * c + ring(n_in, r_in) + ring(n_out, 1.0, ph)
                    sv = np.linalg.svd(phi_matrix(pts, k)[0], compute_uv=False)
                    kap = sv[0] / sv[-1] if sv[-1] > 1e-9 * sv[0] else float('inf')
                    if best is None or kap < best[0]:
                        best = (kap, c, n_in, n_out, float(r_in), float(ph))
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    center = [(0.0, 0.0)]
    lattice = centered([(i, j) for i in range(4) for j in range(4 - i)])
    cases = [
        ('equilateral triangle', ring(3), 1),
        ('pentagon plus center', ring(5) + center, 2),
        ('regular hexagon', ring(6), 2),
        ('hexagon plus center', ring(6) + center, 2),
        ('9-ring plus center', ring(9) + center, 3),
        ('triangular lattice 1+2+3+4', lattice, 3),
        ('two rotated pentagons (r = 0.5, 1)', ring(5, 0.5) + ring(5, 1.0, math.pi / 5), 3),
        ('3 inner + 6 outer + center', ring(3, 0.5) + ring(6, 1.0, math.pi / 6) + center, 3),
    ]
    results = [analyze(n, p, k) for n, p, k in cases]

    print(f"{'formation':38} {'k':>2} {'N':>3} {'rank':>5} {'kappa':>9}  survives single loss")
    for r in results:
        kap = 'singular' if r['kappa'] == float('inf') else f"{r['kappa']:.2f}"
        surv = r.get('survives_any_single_loss', '-')
        extra = f"  fatal: {r['fatal_losses']}" if r.get('fatal_losses') else ''
        print(f"{r['formation']:38} {r['k']:>2} {r['N']:>3} {r['rank']:>2}/{r['n_k']:<2} {kap:>9}  {surv}{extra}")
        if 'null_curve' in r:
            print(f"{'':44}null curve: {r['null_curve']}")

    # validation against Draft 11
    pc = next(r for r in results if r['formation'] == 'pentagon plus center')
    assert abs(pc['kappa'] - 8.26) < 0.01, pc['kappa']
    print(f"\ncheck: pentagon plus center kappa = {pc['kappa']:.3f} (Draft 11 reports 8.26)")

    # noise gain against N, k = 2, (N-1)-gon plus center at rho = 1
    sweep = []
    for N in range(6, 16):
        r = analyze(f'{N-1}-gon plus center', ring(N - 1) + center, 2)
        sweep.append({'N': N, 'kappa': r['kappa'], 'gain_q0': r['gain_by_order'][0],
                      'gain_q1': r['gain_by_order'][1], 'gain_q2': r['gain_by_order'][2],
                      'survives_any_single_loss': r['survives_any_single_loss'],
                      'fatal_losses': r['fatal_losses']})
    Ns = np.array([s['N'] for s in sweep], dtype=float)
    slopes = {q: float(np.polyfit(np.log(Ns), np.log([s[f'gain_q{q}'] for s in sweep]), 1)[0])
              for q in (0, 1, 2)}
    print(f"\n{'N':>3} {'kappa':>7} {'gain q0':>8} {'gain q1':>8} {'gain q2':>8}  survives single loss")
    for s in sweep:
        print(f"{s['N']:>3} {s['kappa']:>7.2f} {s['gain_q0']:>8.3f} {s['gain_q1']:>8.3f} "
              f"{s['gain_q2']:>8.3f}  {s['survives_any_single_loss']} (fatal: {s['fatal_losses']})")
    print('log-log slope of gain against N:', {q: round(v, 3) for q, v in slopes.items()})

    # same sweep for center + two rings, which is not confined to one circle
    sweep2 = []
    for N in range(7, 16):
        r = analyze(f'center + two rings, N={N}', two_ring(N), 2)
        sweep2.append({'N': N, 'kappa': r['kappa'], 'gain_q0': r['gain_by_order'][0],
                       'gain_q1': r['gain_by_order'][1], 'gain_q2': r['gain_by_order'][2],
                       'survives_any_single_loss': r['survives_any_single_loss'],
                       'fatal_losses': r['fatal_losses']})
    Ns2 = np.array([s['N'] for s in sweep2], dtype=float)
    # fit over N >= 10, where both rings hold at least four robots and
    # kappa has settled; N = 7-9 have a three- or four-robot inner ring
    sel = [s for s in sweep2 if s['N'] >= 10]
    slopes2 = {q: float(np.polyfit(np.log([s['N'] for s in sel]),
                                   np.log([s[f'gain_q{q}'] for s in sel]), 1)[0])
               for q in (0, 1, 2)}
    print(f"\ncenter + two rings (inner r = 0.5)")
    print(f"{'N':>3} {'kappa':>7} {'gain q0':>8} {'gain q1':>8} {'gain q2':>8}  survives single loss")
    for s in sweep2:
        print(f"{s['N']:>3} {s['kappa']:>7.2f} {s['gain_q0']:>8.3f} {s['gain_q1']:>8.3f} "
              f"{s['gain_q2']:>8.3f}  {s['survives_any_single_loss']} (fatal: {s['fatal_losses']})")
    print('log-log slope of gain against N (N >= 10):', {q: round(v, 3) for q, v in slopes2.items()})

    b = best_ten()
    print(f"\nbest 10-robot cubic layout found: kappa = {b[0]:.2f}, center = {b[1]}, "
          f"inner {b[2]} at r = {b[4]:.2f}, outer {b[3]}, outer phase = {b[5]:.3f} rad")

    json.dump({'formations': results, 'gain_vs_N_k2_ring_center': sweep,
               'slopes_vs_N_ring_center': slopes, 'gain_vs_N_k2_two_ring': sweep2,
               'slopes_vs_N_two_ring': slopes2,
               'best_ten_cubic': dict(zip(['kappa', 'center', 'n_in', 'n_out', 'r_in', 'phase'], b))},
              open(os.path.join(OUT, 'geometry.json'), 'w'), indent=2, default=str)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    N2 = np.array([s['N'] for s in sel], dtype=float)
    for q, mk in ((0, 'o'), (2, '^')):
        ax.loglog(Ns, [s[f'gain_q{q}'] for s in sweep], mk + '--', color=f'C{q}',
                  label=f'ring + center, order {q}')
        ax.loglog(N2, [s[f'gain_q{q}'] for s in sel], mk + '-', color=f'C{q}',
                  label=f'two rings + center, order {q}')
        ax.loglog(N2, sel[0][f'gain_q{q}'] * (N2 / N2[0]) ** -0.5, 'k:', lw=1,
                  label=r'$N^{-1/2}$' if q == 0 else None)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator(Ns))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(n)) for n in Ns]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_xlabel('robots N')
    ax.set_ylabel(r'noise gain at $\rho = 1$')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'geometry_gain_vs_N.png'), dpi=200, bbox_inches='tight')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
