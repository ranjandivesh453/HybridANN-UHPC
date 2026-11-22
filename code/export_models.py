"""Train the ANN, ANN-ASO, ANN-AEO and ANN-WDO ensembles (ten networks each) and export them for the web tool
and MATLAB.

Usage (from the repository root):  python code/export_models.py
Writes: model.json, model.js (web tool), matlab/UHPC_models.mat
Runtime: under one minute.

The hidden layers, activations, metaheuristic iterations and LM epochs are those selected by grouped 5-fold CV in
reproduce.py (Table 8); hybrids search the initial weights with population 50 before Levenberg-Marquardt training. Each model is an ensemble of ten networks (seeds 1000-1009) trained on the
training mixtures; its prediction is the member mean. Two split-conformal 95% intervals are calibrated on the
validation mixtures: constant (mean +/- q95) and normalized (mean +/- q95_norm * (sd + gamma)), where sd is the
member standard deviation and gamma its median on the validation mixtures.
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree, ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from uhpc_ann import Model, unpack  # noqa: E402

IN = ['W', 'C', 'SF', 'GGBS', 'T', 'Age', 'StF', 'PPF']
UNITS = ['L/m3', 'kg/m3', 'kg/m3', 'kg/m3', 'degC', 'days', 'kg/m3', 'kg/m3']
CONFIG = {'ANN': ('ANN', (10,), 'logsig', 500, 300), 'ANN_ASO': ('ASO', (15, 15), 'logsig', 300, 300),
          'ANN_AEO': ('AEO', (5,), 'logsig', 500, 300), 'ANN_WDO': ('WDO', (5, 5), 'tansig', 300, 300)}   # (method, hidden layers, activation, iterations, LM epochs)
LABEL = {'ANN': 'ANN (LM)', 'ANN_ASO': 'ANN-ASO', 'ANN_AEO': 'ANN-AEO', 'ANN_WDO': 'ANN-WDO'}

# materials: densities (specific gravity), A1-A3 emission factors (kg CO2/kg; Alsomiri et al. 2026) and
# unit costs (USD/kg; Wakjira & Goshu 2026, Table 2; coarse aggregate = sand; PP fibre assumed)
MATERIALS = {
    'keys': ['C', 'GGBS', 'SF', 'W', 'SP', 'CA', 'FA', 'StF', 'PPF'],
    'names': ['OPC', 'GGBS', 'Silica fume', 'Water', 'Superplasticizer', 'Coarse aggregate', 'Fine aggregate', 'Steel fibre', 'PP fibre'],
    'density': {'C': 3.15, 'GGBS': 2.91, 'SF': 2.20, 'W': 1.0, 'SP': 1.077, 'CA': 2.82, 'FA': 2.51, 'StF': 7.8, 'PPF': 0.91},
    'co2': {'C': 0.83, 'GGBS': 0.019, 'SF': 0.014, 'W': 0.0, 'SP': 0.72, 'CA': 0.043, 'FA': 0.024, 'StF': 2.68, 'PPF': 1.6},
    'cost': {'C': 0.082, 'GGBS': 0.100, 'SF': 0.800, 'W': 0.001, 'SP': 3.400, 'CA': 0.009, 'FA': 0.009, 'StF': 1.000, 'PPF': 3.000},
    'co2_source': 'Alsomiri et al. (2026), Eng. Proc. 161, 24 (A1-A3, kg CO2/kg)',
    'cost_source': 'Wakjira & Goshu (2026), Mater. Today Adv. 32, 100944, Table 2 (USD/kg); coarse aggregate taken equal to sand; PP fibre assumed (3.0 USD/kg)',
}


MAT_ORDER = ['C', 'SF', 'GGBS', 'FA', 'CA', 'SP', 'StF', 'PPF', 'W']  # order used by matlab/UHPC_inverse_design.m


def conformal_q(res, alpha=0.05):
    r = np.sort(np.abs(res)); k = math.ceil((len(r) + 1) * (1 - alpha))
    return float(r[min(k, len(r)) - 1])


def main():
    A = pd.read_excel(ROOT / 'data' / 'Supplementary_Data_S1_Cleaned_UHPC_Database.xlsx', sheet_name='Cleaned_Data')
    X, y, g, sub = A[IN].values.astype(float), A.CS.values.astype(float), A.Mix_ID.values, A.Subset.values
    TR, VA, TE = sub == 'Training', sub == 'Validation', sub == 'Test'
    nets = {}; ens_val = {}
    for key, (meth, H, act, T, E) in CONFIG.items():
        members, P = [], []
        for s in range(10):
            M = Model(meth, H, act, 1000 + s, T=T, lm_epochs=E).fit(X[TR], y[TR], g[TR]); P.append(M.predict(X))
            members.append(dict(seed=1000 + s, layers=[dict(W=W.tolist(), b=b.tolist()) for W, b in unpack(M.theta, M.n, M.H)],
                                xmin=M.sx.lo.tolist(), xmax=M.sx.hi.tolist(), ymin=float(M.sy.lo[0]), ymax=float(M.sy.hi[0])))
        P = np.array(P); p = P.mean(0); sd = P.std(0, ddof=1); gam = float(np.median(sd[VA]))
        rm = lambda k: float(np.sqrt(((y[k] - p[k]) ** 2).mean()))
        ens_val[key] = rm(VA)
        nets[key] = dict(label=LABEL[key], method=meth, hidden=list(H), activation=act, iterations=T if meth != 'ANN' else 0, lm_epochs=E, members=members,
                         q95=conformal_q(y[VA] - p[VA]), q95_norm=conformal_q((y[VA] - p[VA]) / (sd[VA] + gam)), gamma=gam,
                         rmse=dict(train=rm(TR), validation=rm(VA), test=rm(TE)))
        print(f"{key:8s} H={'-'.join(map(str, H)):6s} {act} T={T} E={E}  ensemble RMSE val {rm(VA):.2f} test {rm(TE):.2f}  q95 {nets[key]['q95']:.2f} q95_norm {nets[key]['q95_norm']:.3f} gamma {gam:.3f}")
    # applicability domain
    lo, hi = X[TR].min(0), X[TR].max(0); Z = (X[TR] - lo) / (hi - lo)
    d, _ = cKDTree(Z).query(Z, k=2); thr = float(np.percentile(d[:, 1], 95))
    plain = A[A.W == 158].groupby('Mixture').first()
    hp = plain[['GGBS', 'SF']].drop_duplicates().values; hull = ConvexHull(hp)
    rho = MATERIALS['density']
    vtot = float(np.mean([sum(r[k] / rho[k] for k in rho) for _, r in plain.iterrows()]))
    domain = dict(binder=740.0, sp=8.88, train_min=lo.tolist(), train_max=hi.tolist(), nn_threshold=thr,
                  train_norm=np.unique(np.round(Z, 6), axis=0).tolist(),
                  hull_GGBS_SF=hp[hull.vertices].tolist(), V_total=vtot, FA_over_CA=float((plain.FA / plain.CA).mean()),
                  cement_min=float(A.C.min()), fibre_binder={'SF': 88.8, 'GGBS': 296.0},
                  StF_range=[39.0, 78.0], PPF_range=[1.82, 9.1], water_plain=158.0, water_fibre=148.0)
    records = [[*map(float, r[IN]), float(r.CS), r.Subset, r.Mixture] for _, r in A.iterrows()]
    mixtures = [dict(name=r.Mixture, W=float(r.W), SF=float(r.SF), GGBS=float(r.GGBS), StF=float(r.StF), PPF=float(r.PPF))
                for _, r in A.groupby('Mixture').first().reset_index().iterrows()]
    selected = min(ens_val, key=ens_val.get)                            # lowest ensemble validation RMSE
    print('selected model:', selected, {k: round(v, 3) for k, v in ens_val.items()})
    out = dict(name='HybridANN-UHPC', version='3.0.0', inputs=IN, units=UNITS, selected=selected, ensemble_validation_rmse=ens_val,
               networks=nets, domain=domain, materials=MATERIALS, records=records, mixtures=mixtures)
    (ROOT / 'model.json').write_text(json.dumps(out, separators=(',', ':')))
    (ROOT / 'model.js').write_text('// generated by code/export_models.py - do not edit\nvar UHPC_MODEL = ' + json.dumps(out, separators=(',', ':')) + ';\n')
    try:
        from scipy.io import savemat
        def cell(items):
            c = np.empty(len(items), dtype=object)
            for i, it in enumerate(items): c[i] = it
            return c
        mat = {k: dict(members=[dict(W=cell([np.array(l['W']) for l in m['layers']]), b=cell([np.array(l['b'])[:, None] for l in m['layers']]),
                                     xmin=np.array(m['xmin'])[None], xmax=np.array(m['xmax'])[None], ymin=m['ymin'], ymax=m['ymax']) for m in v['members']],
                       activation=v['activation'], q95=v['q95'], q95_norm=v['q95_norm'], gamma=v['gamma']) for k, v in nets.items()}
        dom = dict(inputs=np.array(IN, dtype=object), train_min=lo[None], train_max=hi[None], train_norm=Z, nn_threshold=thr,
                   hull_GGBS_SF=hp[hull.vertices], V_total=vtot, FA_over_CA=domain['FA_over_CA'],
                   EF_names=np.array(MAT_ORDER, dtype=object), EF=np.array([MATERIALS['co2'][k] for k in MAT_ORDER]),
                   cost=np.array([MATERIALS['cost'][k] for k in MAT_ORDER]))
        savemat(ROOT / 'matlab' / 'UHPC_models.mat', dict(models=mat, domain=dom, best_model=selected))
    except Exception as e:  # pragma: no cover
        print('MATLAB export skipped:', e)
    print('wrote model.json, model.js and matlab/UHPC_models.mat')


if __name__ == '__main__':
    main()
