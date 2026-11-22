"""Constraint-consistent permutation importance of the six design factors (Figure 19, Section 4.7).
Usage (from the repository root):  python code/importance.py      (about 1 min)
Cement and water are not independent inputs: C = 740 - SF - GGBS and W = 148 L/m3 with fibres, 158 L/m3 without.
Each factor is permuted across all records, C and W are recomputed, and steel/PP fibre are permuted only among
records without the other fibre type. Output: results/Fig19_importance.csv"""
import json, os, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = Path(__file__).resolve().parents[1]
IN = ['W', 'C', 'SF', 'GGBS', 'T', 'Age', 'StF', 'PPF']; ix = {c: IN.index(c) for c in IN}
FACT = {'Curing age': 'Age', 'Curing temperature': 'T', 'Silica fume': 'SF', 'GGBS': 'GGBS', 'Steel fibre': 'StF', 'PP fibre': 'PPF'}

def repair(Z):
    Z = Z.copy(); Z[:, ix['C']] = 740.0 - Z[:, ix['SF']] - Z[:, ix['GGBS']]
    Z[:, ix['W']] = np.where((Z[:, ix['StF']] > 0) | (Z[:, ix['PPF']] > 0), 148.0, 158.0); return Z

def factor_importance(predict, X, y, n=200, seed=1):
    """mean, 2.5% and 97.5% percentiles of the RMSE increase for each design factor"""
    rng = np.random.default_rng(seed); base = np.sqrt(((y - predict(X)) ** 2).mean()); out = {}
    for name, c in FACT.items():
        j = ix[c]; other = {'StF': 'PPF', 'PPF': 'StF'}.get(c)
        pool = np.arange(len(X)) if other is None else np.where(X[:, ix[other]] == 0)[0]
        inc = []
        for _ in range(n):
            Z = X.copy(); Z[pool, j] = X[rng.permutation(pool), j]; inc.append(np.sqrt(((y - predict(repair(Z))) ** 2).mean()) - base)
        out[name] = (np.mean(inc), np.percentile(inc, 2.5), np.percentile(inc, 97.5))
    return out

def measured_effects(A):
    """mean range of measured CS across the levels of each factor with all other factors fixed"""
    keys = {'Age': ['W', 'SF', 'GGBS', 'T', 'StF', 'PPF'], 'T': ['W', 'SF', 'GGBS', 'Age', 'StF', 'PPF'], 'SF': ['GGBS', 'T', 'Age', 'StF', 'PPF'], 'GGBS': ['SF', 'T', 'Age', 'StF', 'PPF']}
    plain = A[(A.StF == 0) & (A.PPF == 0)]; out = {}
    for name, c in FACT.items():
        if c in keys:
            d = A if c == 'Age' else plain
            r = d.groupby(keys[c]).agg(n=(c, 'nunique'), r=('CS', lambda s: s.max() - s.min())); out[name] = r[r.n >= 2].r.mean()
        else:   # fibre dosage series of G40/SF12 cured at 28 C
            o = 'PPF' if c == 'StF' else 'StF'; d = A[(A[o] == 0) & (A['T'] == 28) & (A.SF == 88.8) & (A.GGBS == 296)]
            out[name] = d.groupby('Age').CS.agg(lambda s: s.max() - s.min()).mean()
    return out

def ann_predict(ens):
    """prediction function of an exported ten-network ensemble (member mean)"""
    act = ens['activation']; f = np.tanh if act == 'tansig' else (lambda z: 1 / (1 + np.exp(-z)))
    def one(net, Z):
        lo, hi = np.array(net['xmin']), np.array(net['xmax']); a = 2 * (Z - lo) / (hi - lo) - 1
        for k, l in enumerate(net['layers']):
            a = a @ np.array(l['W']).T + np.array(l['b'])
            if k < len(net['layers']) - 1: a = f(a)
        return (a[..., 0] + 1) / 2 * (net['ymax'] - net['ymin']) + net['ymin']
    return lambda Z: np.mean([one(n, Z) for n in ens['members']], 0)


if __name__ == '__main__':
    os.chdir(ROOT)
    A = pd.read_excel(ROOT / 'data' / 'Supplementary_Data_S1_Cleaned_UHPC_Database.xlsx', sheet_name='Cleaned_Data')
    X, y, TR = A[IN].values.astype(float), A.CS.values.astype(float), (A.Subset == 'Training').values
    M = json.load(open(ROOT / 'model.json')); sel = M['selected']
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(len(IN))) + WhiteKernel(1e-2)
    gpr = make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0)).fit(X[TR], y[TR])
    ia = factor_importance(ann_predict(M['networks'][sel]), X, y); ig = factor_importance(gpr.predict, X, y); me = measured_effects(A)
    rows = [[f, *ia[f], ig[f][0], me[f]] for f in FACT]
    df = pd.DataFrame(rows, columns=['Factor', f'{sel} RMSE increase (MPa)', 'CI low', 'CI high', 'GPR RMSE increase (MPa)', 'Measured effect (MPa)'])
    df.to_csv('results/Fig19_importance.csv', index=False); print(df.round(2).to_string(index=False))
    rho = spearmanr(df.iloc[:, 1], df.iloc[:, 5]); print(f'Spearman rho ({sel} vs measured) = {rho.statistic:.2f} (p = {rho.pvalue:.3f})')
