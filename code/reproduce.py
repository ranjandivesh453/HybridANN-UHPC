"""Reproduce all model results of the revised manuscript from Supplementary Data S1.
Usage (from the repository root):  python code/reproduce.py      (several hours on 2 CPU cores)
Protocol: hyperparameters of each ANN are selected by grouped 5-fold CV on the training mixtures in two stages
(1: one or two hidden layers, neurons and activation at 500 iterations / 300 epochs; 2: metaheuristic iterations
300/500/1000 and LM epochs 300/500/1000). Every ANN model is a ten-network ensemble (seeds 1000-1009); hybrids search
the initial weights with the metaheuristic (population 50, fitness = mean of inner-training and inner-validation MSE)
before Levenberg-Marquardt training; the selected model has the lowest ensemble validation RMSE.
Outputs: results/*.csv (Tables 8, 13 and 14 and the importance values), results/summary.json
The mix optimization (Tables 15-16, Figure 24) is reproduced by code/optimize_mix.py."""
import json, itertools, os, sys, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
from scipy.stats import beta
from scipy.spatial import cKDTree, Delaunay
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code')); os.chdir(ROOT)
from uhpc_ann import Model  # noqa: E402

A = pd.read_excel(ROOT / 'data' / 'Supplementary_Data_S1_Cleaned_UHPC_Database.xlsx', sheet_name='Cleaned_Data')
IN = ['W', 'C', 'SF', 'GGBS', 'T', 'Age', 'StF', 'PPF']
X, y, g, sub = A[IN].values.astype(float), A.CS.values.astype(float), A.Mix_ID.values, A.Subset.values
TR, VA, TE = sub == 'Training', sub == 'Validation', sub == 'Test'
METHODS = ['ANN', 'ASO', 'AEO', 'WDO']; os.makedirs('results', exist_ok=True)

def m4(t, p):
    e = t - p
    return dict(R2=1 - e @ e / ((t - t.mean()) ** 2).sum(), RMSE=np.sqrt((e ** 2).mean()),
                MAE=np.abs(e).mean(), MAPE=100 * np.mean(np.abs(e / t)))
def job(a):
    m, H, act, T, E, seed, tri, tei = a
    M = Model(m, H, act, seed, T=T, lm_epochs=E).fit(X[tri], y[tri], g[tri]); return M.predict(X[tei]), M.time, M.nfe
def pmap(jobs):
    with ProcessPoolExecutor(2) as ex: return list(ex.map(job, jobs, chunksize=2))
def cv_run(jobs, keyf):
    se = {}
    for a, (p, _, _) in zip(jobs, pmap(jobs)): se.setdefault(keyf(a), []).append((y[a[-1]] - p) ** 2)
    return {k: np.sqrt(np.concatenate(v).mean()) for k, v in se.items()}

if __name__ == '__main__':
    # 1) hyperparameters: grouped 5-fold CV on training mixtures, two seeds per fold
    idx = np.where(TR)[0]; folds = [(idx[a], idx[b]) for a, b in GroupKFold(5).split(X[idx], y[idx], g[idx])]
    ARCH = [(5,), (10,), (15,), (20,), (25,), (30,), (5, 5), (10, 10), (15, 15), (20, 20)]
    jobs = [(m, H, act, 500, 300, s * 10 + k, f[0], f[1]) for m in METHODS for H in ARCH for act in ['tansig', 'logsig'] for k, f in enumerate(folds) for s in range(2)]
    cv1 = cv_run(jobs, lambda a: (a[0], a[1], a[2]))                      # stage 1: architecture
    best1 = {m: min([k for k in cv1 if k[0] == m], key=lambda k: cv1[k]) for m in METHODS}
    jobs = [(m, best1[m][1], best1[m][2], T, E, s * 10 + k, f[0], f[1]) for m in METHODS for T in ([500] if m == 'ANN' else [300, 500, 1000])
            for E in [300, 500, 1000] for k, f in enumerate(folds) for s in range(2)]
    cv = cv_run(jobs, lambda a: (a[0], a[1], a[2], a[3], a[4]))             # stage 2: iterations and epochs
    best = {m: min([k for k in cv if k[0] == m], key=lambda k: cv[k]) for m in METHODS}
    # 2) ten-network ensembles on the training mixtures
    alli = np.arange(len(y)); tri = np.where(TR)[0]
    jobs = [(m, *best[m][1:], 1000 + s, tri, alli) for m in METHODS for s in range(10)]
    runs = {}
    for (m, *_), (p, t, nfe) in zip(jobs, pmap(jobs)):
        runs.setdefault(m, []).append(dict(pred=p, time=t, nfe=nfe, **{s: m4(y[k], p[k]) for s, k in [('Training', TR), ('Validation', VA), ('Test', TE)]}))
    P = {m: np.mean([r['pred'] for r in runs[m]], 0) for m in METHODS}; SD = {m: np.std([r['pred'] for r in runs[m]], 0, ddof=1) for m in METHODS}
    sel = min(METHODS, key=lambda m: m4(y[VA], P[m][VA])['RMSE'])
    t13 = [[m + ' ensemble', s] + [f"{m4(y[k], P[m][k])[c]:.3f}" for c in ['R2', 'RMSE', 'MAE', 'MAPE']] +
           [f"{np.mean([r[s]['RMSE'] for r in runs[m]]):.2f} ± {np.std([r[s]['RMSE'] for r in runs[m]], ddof=1):.2f}"]
           for m in METHODS for s, k in [('Training', TR), ('Validation', VA), ('Test', TE)]]
    # 3) baselines
    def mk(name):
        if name == 'MLR': return make_pipeline(StandardScaler(), LinearRegression())
        if name == 'RF': return RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
        k = ConstantKernel(1.0) * RBF(length_scale=np.ones(len(IN))) + WhiteKernel(1e-2)
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0))
    def fx(Z, name):
        Z = Z.copy()
        if name == 'MLR': Z[:, IN.index('Age')] = np.log(Z[:, IN.index('Age')])
        return Z
    BP, BL, B5 = {}, {}, {}
    for name in ['MLR', 'RF', 'GPR']:
        BP[name] = mk(name).fit(fx(X[TR], name), y[TR]).predict(fx(X, name))
        for s, k in [('Training', TR), ('Validation', VA), ('Test', TE)]: t13.append([name, s] + [f"{m4(y[k], BP[name][k])[c]:.3f}" for c in ['R2', 'RMSE', 'MAE', 'MAPE']] + ['-'])
        pl = np.zeros(len(y))
        for mx in np.unique(g):
            o = g == mx; pl[o] = mk(name).fit(fx(X[~o], name), y[~o]).predict(fx(X[o], name))
        BL[name] = pl; p5 = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=0).split(X): p5[b] = mk(name).fit(fx(X[a], name), y[a]).predict(fx(X[b], name))
        B5[name] = p5
    pd.DataFrame(t13, columns=['Model', 'Subset', 'R2', 'RMSE', 'MAE', 'MAPE', 'Single-network RMSE (mean ± SD)']).to_csv('results/Table13_performance.csv', index=False)
    # 4) LOMO and record-level 5-fold CV for the ANNs (ten-network ensembles)
    mixes = np.unique(g)
    jobs = [(m, *best[m][1:], 2000 + s * 100 + k, np.where(g != mx)[0], np.where(g == mx)[0]) for m in METHODS for s in range(10) for k, mx in enumerate(mixes)]
    L = {}
    for (m, H, act, T_, E_, seed, tri_, tei), (p, _, _) in zip(jobs, pmap(jobs)): L.setdefault((m, (seed - 2000) // 100), np.zeros(len(y)))[tei] = p
    jobs = [(m, *best[m][1:], 3000 + s * 100 + k, a, b) for m in METHODS for s in range(10) for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(X))]
    R5 = {}
    for (m, H, act, T_, E_, seed, a, b), (p, _, _) in zip(jobs, pmap(jobs)): R5.setdefault((m, (seed - 3000) // 100), np.zeros(len(y)))[b] = p
    LE = {m: np.mean([L[(m, s)] for s in range(10)], 0) for m in METHODS}; R5E = {m: np.mean([R5[(m, s)] for s in range(10)], 0) for m in METHODS}
    # 5) split-conformal intervals (validation calibration) on the test mixtures: normalized by the ensemble spread for the ANNs
    def conformal(p, sig=None):
        sig = np.ones(len(y)) if sig is None else sig
        r = np.sort(np.abs(y[VA] - p[VA]) / sig[VA]); n = len(r); q = r[min(int(np.ceil((n + 1) * 0.95)), n) - 1]
        lo, hi = p[TE] - q * sig[TE], p[TE] + q * sig[TE]; t = y[TE]; c = np.mean((t >= lo) & (t <= hi)); k = int(round(c * len(t)))
        ci = (beta.ppf(0.025, k, len(t) - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, len(t) - k) if k < len(t) else 1.0)
        return q, c, ci, (hi - lo).mean(), ((hi - lo) + 40 * np.clip(lo - t, 0, None) + 40 * np.clip(t - hi, 0, None)).mean()
    t14 = []
    for m in METHODS:
        le = m4(y, LE[m]); sing = [m4(y, L[(m, s)])['RMSE'] for s in range(10)]; q, c, ci, w, ws = conformal(P[m], SD[m] + np.median(SD[m][VA]))
        t14.append([m + ' ensemble', f"{le['R2']:.3f}", f"{le['RMSE']:.2f} ({np.mean(sing):.2f} ± {np.std(sing, ddof=1):.2f})", f"{le['MAE']:.2f}", f"{m4(y, R5E[m])['RMSE']:.2f}",
                    f"{100*c:.1f} [{100*ci[0]:.1f}-{100*ci[1]:.1f}]", f"{w:.2f}", f"{ws:.2f}"])
    for name in ['MLR', 'RF', 'GPR']:
        lm = m4(y, BL[name]); q, c, ci, w, ws = conformal(BP[name])
        t14.append([name, f"{lm['R2']:.3f}", f"{lm['RMSE']:.2f}", f"{lm['MAE']:.2f}", f"{m4(y, B5[name])['RMSE']:.2f}", f"{100*c:.1f} [{100*ci[0]:.1f}-{100*ci[1]:.1f}]", f"{w:.2f}", f"{ws:.2f}"])
    pd.DataFrame(t14, columns=['Model', 'LOMO R2', 'LOMO RMSE (single-network mean ± SD)', 'LOMO MAE', 'Record-level 5-fold RMSE', 'PICP [95% CI]', 'MPIW', 'Winkler']).to_csv('results/Table14_LOMO_UQ.csv', index=False)
    # 6) tests: selected ensemble vs others on the test errors, and on the 38 per-mixture LOMO RMSEs
    rng = np.random.default_rng(0); eb = y[TE] - P[sel][TE]; tests = {}
    LP = dict(LE); LP.update(BL); pm_ = lambda p: pd.Series(y - p).groupby(A.Mixture.values).apply(lambda v: np.sqrt((v ** 2).mean())).values
    for m, pm in [(m, P[m][TE]) for m in METHODS if m != sel] + [(b, BP[b][TE]) for b in ['MLR', 'RF', 'GPR']]:
        eo = y[TE] - pm; d = []
        for _ in range(2000):
            i = rng.integers(0, len(eb), len(eb)); d.append(np.sqrt((eb[i] ** 2).mean()) - np.sqrt((eo[i] ** 2).mean()))
        tests[m] = dict(wilcoxon_p=float(stats.wilcoxon(np.abs(eb), np.abs(eo)).pvalue), rmse_diff_ci=list(np.percentile(d, [2.5, 97.5])),
                        lomo_wilcoxon_p=float(stats.wilcoxon(pm_(LP[sel]), pm_(LP[m])).pvalue))
    # 7) constraint-consistent permutation importance of the design factors (all records; see code/importance.py)
    from importance import factor_importance, measured_effects, FACT
    from uhpc_ann import Ensemble
    M = Ensemble([Model(sel, best[sel][1], best[sel][2], 1000 + s, T=best[sel][3], lm_epochs=best[sel][4]).fit(X[TR], y[TR], g[TR]) for s in range(10)])
    gpr = mk('GPR').fit(fx(X[TR], 'GPR'), y[TR])
    ia = factor_importance(M.predict, X, y); ig = factor_importance(lambda Z: gpr.predict(fx(Z, 'GPR')), X, y); me = measured_effects(A)
    pd.DataFrame([[f, *ia[f], ig[f][0], me[f]] for f in FACT], columns=['Factor', f'{sel} RMSE increase (MPa)', 'CI low', 'CI high', 'GPR RMSE increase (MPa)', 'Measured effect (MPa)']).to_csv('results/Fig19_importance.csv', index=False)
    t8 = [[m, '-'.join(map(str, best[m][1])), best[m][2], best[m][3] if m != 'ANN' else '-', best[m][4], f"{cv[best[m]]:.2f}", f"{np.mean([r['time'] for r in runs[m]]):.2f}", f"{np.mean([r['nfe'] for r in runs[m]]):.0f}"] for m in METHODS]
    pd.DataFrame(t8, columns=['Model', 'Hidden neurons', 'Activation', 'Metaheuristic iterations', 'LM epochs', 'Grouped CV RMSE', 'Time per network (s)', 'Function evaluations per network']).to_csv('results/Table8_configuration.csv', index=False)
    json.dump(dict(selected=sel, tests=tests), open('results/summary.json', 'w'), indent=1, default=float)
    print('done; selected model:', sel)
