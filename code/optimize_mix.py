"""Tri-objective mix optimization: maximize 28-day strength (selected network), minimize embodied CO2 (A1-A3) and
minimize material cost, restricted to the experimental domain (NSGA-III, pymoo).

Usage (from the repository root, after code/export_models.py):  python code/optimize_mix.py
Writes: pareto_data.js (web tool), results/pareto_solutions.csv, results/representative_solutions.csv,
        results/inverse_design.csv, results/optimization_summary.json, assets/fig_pareto.png
Runtime: about 2 min.

Decision variables: SF (0-133.2 kg/m3), GGBS (0-370 kg/m3), fibre type (none / steel / PP) and fibre dosage.
Fixed: binder 740 kg/m3, curing 28 degC, age 28 days, SP 8.88 kg/m3, water 158 (plain) or 148 L/m3 (fibre),
aggregates from the absolute-volume equation. Constraints: (GGBS, SF) inside the convex hull of the tested plain
compositions; fibres only with the tested G40/SF12 binder (StF 39-78 or PPF 1.82-9.1 kg/m3); cement >= 281.2
kg/m3; nearest-neighbour distance to the training data <= 95th percentile of within-training distances;
lower 95% prediction bound >= 120 MPa (normalized split-conformal interval of the selected ensemble).
Also writes results/improvement_vs_tested.csv: for each tested 28-degC mixture, the lowest-CO2 and the lowest-cost
Pareto solution with at least the same predicted 28-day strength.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree, ConvexHull
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads((ROOT / 'model.json').read_text())
IN = MODEL['inputs']; DOM = MODEL['domain']; MAT = MODEL['materials']
RHO, EF, COST = MAT['density'], MAT['co2'], MAT['cost']
T0, AGE0, FC_MIN, N_SEEDS, POP, GEN = 28.0, 28.0, 120.0, 5, 300, 500


def net_predict(key, X):
    """ensemble mean and member standard deviation"""
    e = MODEL['networks'][key]; P = []
    for n in e['members']:
        a = 2 * (X - np.array(n['xmin'])) / (np.array(n['xmax']) - np.array(n['xmin'])) - 1
        for k, l in enumerate(n['layers']):
            a = a @ np.array(l['W']).T + np.array(l['b'])
            if k < len(n['layers']) - 1: a = np.tanh(a) if e['activation'] == 'tansig' else 1 / (1 + np.exp(-a))
        P.append((a[..., 0] + 1) / 2 * (n['ymax'] - n['ymin']) + n['ymin'])
    P = np.array(P); return P.mean(0), P.std(0, ddof=1)


def half_width(sd):
    return QN * (sd + GAM)


A = pd.read_excel(ROOT / 'data' / 'Supplementary_Data_S1_Cleaned_UHPC_Database.xlsx', sheet_name='Cleaned_Data')
LO, HI = np.array(DOM['train_min']), np.array(DOM['train_max'])
nz = lambda Z: (Z - LO) / (HI - LO)
TREE = cKDTree(nz(A.loc[A.Subset == 'Training', IN].values.astype(float))); THR = DOM['nn_threshold']
HEQ = ConvexHull(np.array(DOM['hull_GGBS_SF'])).equations
SEL = MODEL['selected']; QN, GAM = MODEL['networks'][SEL]['q95_norm'], MODEL['networks'][SEL]['gamma']; B, SP0, VTOT, FACA = DOM['binder'], DOM['sp'], DOM['V_total'], DOM['FA_over_CA']


def compose(SF, GG, ft, frac):
    """Mixture (kg/m3) for vectors of SF, GGBS, fibre type (0 none, 1 steel, 2 PP) and dosage fraction."""
    SF = np.asarray(SF, float); GG = np.asarray(GG, float); ft = np.asarray(ft, int); frac = np.asarray(frac, float)
    fib = ft > 0
    SF = np.where(fib, 88.8, SF); GG = np.where(fib, 296.0, GG)           # fibres only with the tested G40/SF12 binder
    StF = np.where(ft == 1, 39.0 + frac * 39.0, 0.0); PPF = np.where(ft == 2, 1.82 + frac * 7.28, 0.0)
    W = np.where(fib, 148.0, 158.0); C = B - SF - GG
    Vb = C / RHO['C'] + GG / RHO['GGBS'] + SF / RHO['SF'] + W + SP0 / RHO['SP'] + StF / RHO['StF'] + PPF / RHO['PPF']
    CA = (VTOT - Vb) / (1 / RHO['CA'] + FACA / RHO['FA']); FA = FACA * CA
    return dict(W=W, SF=SF, GGBS=GG, C=C, SP=np.full_like(SF, SP0), CA=CA, FA=FA, StF=StF, PPF=PPF,
                T=np.full_like(SF, T0), Age=np.full_like(SF, AGE0))


def evaluate(mix):
    X = np.column_stack([mix[k] for k in IN])
    fc, sd = net_predict(SEL, X); hw = half_width(sd)
    co2 = sum(mix[k] * EF[k] for k in EF); cost = sum(mix[k] * COST[k] for k in COST)
    nn = TREE.query(nz(X), k=1)[0]
    gh = (np.column_stack([mix['GGBS'], mix['SF']]) @ HEQ[:, :2].T + HEQ[:, 2]).max(1)
    gh = np.where((mix['StF'] > 0) | (mix['PPF'] > 0), -1.0, gh)
    G = np.column_stack([gh - 1e-6, nn - THR, FC_MIN - (fc - hw), DOM['cement_min'] - mix['C'] - 1e-6])
    return fc, co2, cost, nn, G, hw


class MixProblem(Problem):
    def __init__(self):
        super().__init__(n_var=4, n_obj=3, n_ieq_constr=4, xl=np.zeros(4), xu=np.array([133.2, 370.0, 2.999, 1.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        fc, co2, cost, nn, G, _ = evaluate(compose(x[:, 0], x[:, 1], np.floor(x[:, 2]).astype(int), x[:, 3]))
        out['F'] = np.column_stack([-fc, co2, cost]); out['G'] = G


class FibreRepair(Repair):
    def _do(self, problem, X, **kwargs):
        fib = np.floor(X[:, 2]) > 0; X[fib, 0] = 88.8; X[fib, 1] = 296.0; return X


def frame(mix, keep=None):
    fc, co2, cost, nn, G, hw = evaluate(mix)
    keep = np.all(G <= 1e-6, 1) if keep is None else keep
    P = pd.DataFrame({k: v[keep] for k, v in mix.items()})
    P['fc'], P['lo'], P['hi'] = fc[keep], fc[keep] - hw[keep], fc[keep] + hw[keep]
    P['CO2'], P['cost'], P['nn'] = co2[keep], cost[keep], nn[keep]
    return P


def encode(df):
    ft = np.where(df.StF > 0, 1, np.where(df.PPF > 0, 2, 0))
    fr = np.where(ft == 1, (df.StF - 39) / 39, np.where(ft == 2, (df.PPF - 1.82) / 7.28, 0))
    return ft, np.clip(fr, 0, 1)


def topsis(obj, w, benefit=(True, False, False)):
    R = obj / np.sqrt((obj ** 2).sum(0)); V = R * np.asarray(w, float) / np.sum(w)
    best = np.where(benefit, V.max(0), V.min(0)); worst = np.where(benefit, V.min(0), V.max(0))
    dp = np.linalg.norm(V - best, axis=1); dn = np.linalg.norm(V - worst, axis=1)
    return dn / (dp + dn)


def main():
    tm = A[A['T'] == 28].groupby('Mixture').first().reset_index()
    ft, fr = encode(tm)
    seedX = np.column_stack([tm.SF, tm.GGBS, ft + 0.5, fr])            # tested mixtures seed the population
    ref = get_reference_directions('das-dennis', 3, n_partitions=23)  # 300 reference directions
    fronts = []
    for seed in range(N_SEEDS):
        rs = np.random.default_rng(seed)
        X0 = np.vstack([seedX, rs.uniform([0, 0, 0, 0], [133.2, 370, 2.999, 1], (POP - len(seedX), 4))])
        alg = NSGA3(ref_dirs=ref, pop_size=POP, sampling=X0, crossover=SBX(eta=15, prob=0.9), mutation=PM(eta=20),
                    repair=FibreRepair(), eliminate_duplicates=True)
        r = minimize(MixProblem(), alg, ('n_gen', GEN), seed=seed, verbose=False)
        PX, PG, PF = r.pop.get('X'), r.pop.get('G'), r.pop.get('F'); ok = np.all(PG <= 0, 1)
        fronts.append(PX[ok][NonDominatedSorting().do(PF[ok], only_non_dominated_front=True)])
        print(f'run {seed + 1}/{N_SEEDS}: {len(fronts[-1])} non-dominated feasible solutions', flush=True)
    Xp = np.vstack(fronts)
    P = frame(compose(Xp[:, 0], Xp[:, 1], np.floor(Xp[:, 2]).astype(int), Xp[:, 3]))
    nd = NonDominatedSorting().do(np.column_stack([-P.fc, P.CO2, P.cost]), only_non_dominated_front=True); P = P.iloc[nd]
    # snap solutions within 1 kg/m3 of a tested composition onto it, round, re-evaluate and filter again
    tcomp = tm[['SF', 'GGBS', 'StF', 'PPF']].values.astype(float); V = P[['SF', 'GGBS', 'StF', 'PPF']].values.copy()
    for i in range(len(V)):
        d = np.abs(tcomp - V[i]).max(1)
        if d.min() <= 1.0: V[i] = tcomp[d.argmin()]
    V = pd.DataFrame(V, columns=['SF', 'GGBS', 'StF', 'PPF']).round({'SF': 1, 'GGBS': 1, 'StF': 1, 'PPF': 2}).drop_duplicates()
    ft, fr = encode(V)
    P = frame(compose(V.SF.values, V.GGBS.values, ft, fr))
    nd = NonDominatedSorting().do(np.column_stack([-P.fc, P.CO2, P.cost]), only_non_dominated_front=True)
    P = P.iloc[nd].sort_values(['fc', 'CO2'], ascending=[False, True]).reset_index(drop=True)
    print('Pareto set:', len(P), 'solutions')

    # exhaustive reference front on a fine grid (0.1% SF, 0.25% GGBS; 400 dosages per fibre type)
    sf, gg = np.meshgrid(np.arange(0, 133.2 + 1e-9, 0.74), np.arange(0, 370 + 1e-9, 1.85))
    SFg = np.r_[sf.ravel(), np.zeros(800)]; GGg = np.r_[gg.ravel(), np.zeros(800)]
    ftg = np.r_[np.zeros(sf.size), np.ones(400), 2 * np.ones(400)].astype(int); frg = np.r_[np.zeros(sf.size), np.tile(np.linspace(0, 1, 400), 2)]
    gfc, gco2, gcost, _, gG, _ = evaluate(compose(SFg, GGg, ftg, frg)); feas = np.all(gG <= 0, 1)
    Fg = np.column_stack([-gfc, gco2, gcost])[feas]; Fg = Fg[NonDominatedSorting().do(Fg, only_non_dominated_front=True)]
    Fn = np.column_stack([-P.fc, P.CO2, P.cost]); allF = np.vstack([Fg, Fn]); ideal, nadir = allF.min(0), allF.max(0)
    hv = HV(ref_point=np.full(3, 1.1)); hv_n, hv_g = hv((Fn - ideal) / (nadir - ideal)), hv((Fg - ideal) / (nadir - ideal))
    print(f'hypervolume NSGA-III {hv_n:.4f}, exhaustive grid {hv_g:.4f}, ratio {hv_n / hv_g:.4f}')

    # representative solutions (as in Wakjira & Goshu 2026) + minimum cost + TOPSIS
    obj = P[['fc', 'CO2', 'cost']].values
    o = (obj - obj.min(0)) / (obj.max(0) - obj.min(0)); o[:, 0] = 1 - o[:, 0]
    P['topsis'] = topsis(obj, [1, 1, 1])
    REP = {'Maximum strength': int(P.fc.idxmax()), 'Minimum CO2': int(P.CO2.idxmin()),
           'Best efficiency (max fc/CO2)': int((P.fc / P.CO2).idxmax()), 'Minimum cost': int(P.cost.idxmin()),
           'Balanced (knee)': int(np.argmin(np.linalg.norm(o, axis=1))), 'TOPSIS (equal weights)': int(P.topsis.idxmax())}
    for name, w in [('TOPSIS strength-led (0.6/0.2/0.2)', [.6, .2, .2]), ('TOPSIS carbon-led (0.2/0.6/0.2)', [.2, .6, .2]),
                    ('TOPSIS cost-led (0.2/0.2/0.6)', [.2, .2, .6])]:
        REP[name] = int(np.argmax(topsis(obj, w)))
    t28 = A[(A['T'] == 28) & (A.Age == 28)].groupby('Mixture').first().reset_index(); tt = cKDTree(nz(t28[IN].values.astype(float)))
    rows = []
    for lab, i in REP.items():
        r = P.iloc[i]; X = r[IN].values.astype(float)[None]
        preds = {k: float(net_predict(k, X)[0][0]) for k in MODEL['networks']}
        _, j = tt.query(nz(X), k=1); nb = t28.iloc[j[0]]
        rows.append(dict(solution=lab, id=i + 1, **{k: round(float(r[k]), 2) for k in ['C', 'GGBS', 'SF', 'W', 'StF', 'PPF', 'CA', 'FA', 'fc', 'lo', 'hi', 'CO2', 'cost', 'nn']},
                         topsis_equal=round(float(r.topsis), 3), **{f'pred_{k}': round(v, 2) for k, v in preds.items()},
                         nearest_tested=nb.Mixture, nearest_measured_MPa=float(nb.CS)))
    REPdf = pd.DataFrame(rows)
    inv = []
    for target in [100, 130, 140, 142]:
        ok = P[P.lo >= target].sort_values(['CO2', 'cost'])
        if len(ok):
            r = ok.iloc[0]; _, j = tt.query(nz(r[IN].values.astype(float)[None]), k=1); nb = t28.iloc[j[0]]
            inv.append(dict(target=target, id=int(ok.index[0]) + 1, **{k: round(float(r[k]), 2) for k in ['C', 'GGBS', 'SF', 'W', 'StF', 'PPF', 'CA', 'FA', 'fc', 'lo', 'hi', 'CO2', 'cost', 'nn']},
                            nearest_tested=nb.Mixture, nearest_measured_MPa=float(nb.CS)))
        else:
            inv.append(dict(target=target))
    INV = pd.DataFrame(inv)
    tq = t28.copy(); Xt = tq[IN].values.astype(float); tq['fc_pred'] = net_predict(SEL, Xt)[0]
    tq['CO2'] = sum(tq[k] * EF[k] for k in EF); tq['cost'] = sum(tq[k] * COST[k] for k in COST); imp = []
    for _, r in tq.iterrows():
        ok = P[P.fc >= r.fc_pred - 1e-9]
        if not len(ok): continue
        a, b = ok.sort_values('CO2').iloc[0], ok.sort_values('cost').iloc[0]
        imp.append(dict(mixture=r.Mixture, measured=float(r.CS), predicted=round(float(r.fc_pred), 2), CO2=round(float(r.CO2), 1), cost=round(float(r.cost), 1),
                        best_CO2=round(float(a.CO2), 1), dCO2_pct=round(100 * (a.CO2 / r.CO2 - 1), 1), best_CO2_id=int(a.name) + 1,
                        best_cost=round(float(b.cost), 1), dcost_pct=round(100 * (b.cost / r.cost - 1), 1), best_cost_id=int(b.name) + 1))
    IMPdf = pd.DataFrame(imp)
    plain = P[(P.StF == 0) & (P.PPF == 0)]
    (ROOT / 'results').mkdir(exist_ok=True)
    out = P.copy(); out.insert(0, 'id', np.arange(1, len(P) + 1)); out.round(3).to_csv(ROOT / 'results' / 'pareto_solutions.csv', index=False)
    REPdf.to_csv(ROOT / 'results' / 'representative_solutions.csv', index=False); IMPdf.to_csv(ROOT / 'results' / 'improvement_vs_tested.csv', index=False); INV.to_csv(ROOT / 'results' / 'inverse_design.csv', index=False)
    summary = dict(n_pareto=len(P), hv_nsga=hv_n, hv_grid=hv_g, hv_ratio=hv_n / hv_g, grid_feasible=int(feas.sum()), grid_candidates=int(feas.size),
                   ranges=dict(fc=[P.fc.min(), P.fc.max()], CO2=[P.CO2.min(), P.CO2.max()], cost=[P.cost.min(), P.cost.max()]),
                   max_plain_lower_bound=float(plain.lo.max()), n_steel=int((P.StF > 0).sum()), n_pp=int((P.PPF > 0).sum()),
                   settings=dict(algorithm='NSGA-III (pymoo)', population=POP, generations=GEN, runs=N_SEEDS, sbx_eta=15, sbx_prob=0.9, pm_eta=20,
                                 reference_directions=len(ref), fc_lower_bound_min=FC_MIN, curing_temp=T0, age=AGE0),
                   representatives={k: int(v) + 1 for k, v in REP.items()})
    (ROOT / 'results' / 'optimization_summary.json').write_text(json.dumps(summary, indent=1, default=float))
    # web data
    sols = []
    for i, r in P.iterrows():
        sols.append({'id': i + 1, 'cement': round(r.C, 1), 'ggbs': round(r.GGBS, 1), 'silica_fume': round(r.SF, 1), 'water': round(r.W, 1),
                     'superplasticizer': round(r.SP, 2), 'coarse_agg': round(r.CA, 1), 'fine_agg': round(r.FA, 1),
                     'steel_fiber': round(r.StF, 1), 'pp_fiber': round(r.PPF, 2), 'strength': round(r.fc, 4), 'lower95': round(r.lo, 4),
                     'upper95': round(r.hi, 4), 'co2': round(r.CO2, 4), 'cost': round(r.cost, 4), 'wb_ratio': round(r.W / B, 3),
                     'scm_ratio': round((r.GGBS + r.SF) / B, 3), 'sfc_ratio': round(r.SF / r.C, 3),
                     'vf': round(r.StF / RHO['StF'] / 10 + r.PPF / RHO['PPF'] / 10, 2), 'nn': round(r.nn, 3)})
    meta = dict(representatives={k: int(v) + 1 for k, v in REP.items()}, hv_ratio=round(hv_n / hv_g, 4), n=len(P), q95_norm=QN, gamma=GAM,
                curing_temp=T0, age=AGE0, fc_lower_bound_min=FC_MIN)
    (ROOT / 'pareto_data.js').write_text('// generated by code/optimize_mix.py - do not edit\nvar PARETO_META = ' + json.dumps(meta) +
                                         ';\nvar PARETO_SOLUTIONS = [\n' + ',\n'.join(json.dumps(s, separators=(',', ':')) for s in sols) + '\n];\n')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(REPdf[['solution', 'C', 'GGBS', 'SF', 'StF', 'fc', 'lo', 'hi', 'CO2', 'cost', 'nearest_tested', 'nearest_measured_MPa']].to_string())
    print(INV[['target', 'C', 'GGBS', 'SF', 'StF', 'fc', 'lo', 'hi', 'CO2', 'cost']].to_string())
    try:
        plot(P, REP, A)
    except Exception as e:  # pragma: no cover
        print('figure skipped:', e)


def plot(P, REP, A):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    t = A[(A['T'] == 28) & (A.Age == 28)].groupby('Mixture').first().reset_index()
    t['CO2'] = sum(t[k] * EF[k] for k in EF); t['cost'] = sum(t[k] * COST[k] for k in COST)
    reps = [('Maximum strength', '^', '#b42318', 'Maximum strength'), ('Minimum CO2', 'v', '#1b7f4b', 'Minimum CO$_2$ / best efficiency'),
            ('Minimum cost', 's', '#7a4cc2', 'Minimum cost'), ('Balanced (knee)', 'D', '#c26a00', 'Balanced (knee)'),
            ('TOPSIS (equal weights)', '*', '#1f5fbf', 'TOPSIS (equal weights)')]
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8.5, 'axes.edgecolor': '#555', 'axes.linewidth': 0.6})
    cmap = plt.get_cmap('Blues'); fig = plt.figure(figsize=(7.2, 6.6))
    def tidy(ax): ax.spines[['top', 'right']].set_visible(False)
    ax = fig.add_subplot(2, 2, 1); tidy(ax)
    ax.scatter(t.CO2, t.cost, s=16, facecolor='white', edgecolor='#888', lw=0.7, zorder=2)
    sc = ax.scatter(P.CO2, P.cost, c=P.fc, cmap=cmap, vmin=115, vmax=155, s=6, lw=0, zorder=3)
    for k, m, c, _ in reps:
        r = P.iloc[REP[k]]; ax.scatter(r.CO2, r.cost, marker=m, s=70 if m == '*' else 38, c=c, edgecolor='k', lw=0.5, zorder=5)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label('Predicted 28-day $f_c$ (MPa)', fontsize=7.5); cb.ax.tick_params(labelsize=7)
    ax.set_xlabel('Embodied CO$_2$ (kg/m$^3$)'); ax.set_ylabel('Material cost (USD/m$^3$)'); ax.set_title('(a) Carbon–cost projection of the Pareto front', fontsize=9, loc='left')
    for j, (xk, xl, tt) in enumerate([('CO2', 'Embodied CO$_2$ (kg/m$^3$)', '(b) Strength–carbon projection'), ('cost', 'Material cost (USD/m$^3$)', '(c) Strength–cost projection')]):
        ax = fig.add_subplot(2, 2, 2 + j); tidy(ax)
        ax.scatter(t[xk], t.CS, s=16, facecolor='white', edgecolor='#888', lw=0.7, label='Tested mixtures (measured)', zorder=2)
        ax.scatter(P[xk], P.fc, c=P.fc, cmap=cmap, vmin=115, vmax=155, s=6, lw=0, label='Pareto solutions (predicted)', zorder=3)
        for k, m, c, lab in reps:
            r = P.iloc[REP[k]]; ax.scatter(r[xk], r.fc, marker=m, s=70 if m == '*' else 38, c=c, edgecolor='k', lw=0.5, zorder=5, label=lab)
        ax.axhline(120, color='#999', lw=0.7, ls='--')
        ax.text(ax.get_xlim()[1] if j else 690, 121, '$f_{c,lo}$ ≥ 120 MPa', ha='right', va='bottom', fontsize=7, color='#666')
        ax.set_xlabel(xl); ax.set_ylabel('28-day $f_c$ (MPa)'); ax.set_title(tt, fontsize=9, loc='left'); ax.set_ylim(80, 160)
        if j == 0: H, L = ax.get_legend_handles_labels()
    ax = fig.add_subplot(2, 2, 4); tidy(ax)
    rows = [P.iloc[REP[k]] for k, *_ in reps]; y = np.arange(len(rows))[::-1]; left = np.zeros(len(rows))
    for key, colr, nm in [('C', '#6b6b6b', 'OPC'), ('GGBS', '#2a78d6', 'GGBS'), ('SF', '#e08a1e', 'SF')]:
        v = np.array([r[key] for r in rows]); ax.barh(y, v, left=left, color=colr, height=0.62, edgecolor='white', lw=1, label=nm); left += v
    for yi, r in zip(y, rows):
        s = f"{r.fc:.1f} MPa | {r.CO2:.0f} kg | ${r.cost:.0f}" + (f" | StF {r.StF:.0f} kg" if r.StF > 0 else '')
        ax.text(745, yi, s, va='center', fontsize=6.4)
    ax.set_yticks(y); ax.set_yticklabels(['Max. strength', 'Min. CO$_2$', 'Min. cost', 'Knee', 'TOPSIS']); ax.set_xlim(0, 1260)
    ax.set_xticks([0, 185, 370, 555, 740]); ax.set_xlabel('Binder composition (kg/m$^3$)'); ax.set_title('(d) Representative solutions', fontsize=9, loc='left')
    ax.set_ylim(-0.6, len(rows) + 0.2); ax.legend(fontsize=7, frameon=False, loc='upper left', ncol=3, bbox_to_anchor=(0.0, 1.02), handlelength=1.2)
    fig.tight_layout(w_pad=1.2, h_pad=1.4, rect=(0, 0.075, 1, 1))
    fig.legend(H, L, loc='lower center', ncol=4, fontsize=7, frameon=False, handletextpad=0.3, columnspacing=1.0)
    (ROOT / 'assets').mkdir(exist_ok=True); fig.savefig(ROOT / 'assets' / 'fig_pareto.png', dpi=300); plt.close(fig)
    print('wrote assets/fig_pareto.png')


if __name__ == '__main__':
    main()
