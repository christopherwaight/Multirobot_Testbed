"""Jitter the current Fig. 9 operating point the same way ridge_jitter.py jitters candidates. Run from VF_Robot: venv/bin/python3 experiments/ocean_small_rho/jitter_paper_current.py $PWD"""
import sys, os, numpy as np, pandas as pd, multiprocessing as mp
sys.path.insert(0, os.path.join(sys.argv[1], 'experiments', 'ocean_small_rho'))
import ridge_jitter as R, ocean_ridge_mc as M
if __name__ == "__main__":
    c = dict(start_lat=34.38493548452217, start_lon=-120.40546793817956, scale=0.934190927524684,
             heading_deg=335.08707973270054, gain=1.940386764778898, v_max=0.0332630274329872)
    rng = np.random.default_rng(7); jobs = [(0, 0, *c.values())]
    for j in range(1, 101):
        r, th = np.sqrt(rng.uniform()), rng.uniform(0, 2*np.pi)
        jobs.append((0, j, c['start_lat'] + r*np.sin(th)/M.KLA, c['start_lon'] + r*np.cos(th)/M.KLO,
                     c['scale']*rng.uniform(.95,1.05), c['heading_deg']+rng.uniform(-15,15),
                     c['gain']*rng.uniform(.95,1.05), c['v_max']*rng.uniform(.95,1.05)))
    with mp.Pool(4, initializer=R._init) as p:
        J = R.add_rank(pd.DataFrame([r for r in p.map(R._job, jobs) if r is not None]))
    J.to_csv(os.path.join(sys.argv[1], 'experiments', 'ocean_small_rho', 'ridge_jitter_paper_current.csv'), index=False)
    j = J[J.jitter > 0]
    print('nominal rank', J[J.jitter == 0]['rank'].iloc[0], '| jitter median rank', j['rank'].median(),
          'frac finite', np.isfinite(j['rank']).mean(), '| med ride D', j.ride_D.median(), 's1', j.ride_s1.median(),
          '| frac rank < nominal', (j['rank'] < J[J.jitter == 0]['rank'].iloc[0]).mean())
