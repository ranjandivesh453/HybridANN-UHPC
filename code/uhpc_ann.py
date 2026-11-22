"""Feed-forward ANN (one or two hidden layers) trained by Levenberg-Marquardt (LM), or initialized by ASO / AEO / WDO and then trained by LM.
All computations in the [-1, 1] normalized space (MATLAB mapminmax convention), fitted on the training data.
Protocol (identical LM budget for every ANN): 20% of the training mixtures are held out as an inner validation set;
LM runs for a fixed number of epochs (tuned: 300, 500 or 1000; or until it cannot reduce the error further) and the
weights with the lowest inner-validation error are kept. Hybrid models first search the weight space with the
metaheuristic (population 50, tuned number of iterations: 300, 500 or 1000; fitness = mean of the inner-training and
inner-validation MSE) and use the best solution as the LM starting point. Learning curves can be recorded (record=True).
Ensemble: mean of independently trained networks (class Ensemble)."""
import numpy as np, time

# ---------------------------------------------------------------- scaling
class MinMax:
    def fit(self, a):
        a = np.asarray(a, float); self.lo = a.min(0); self.hi = a.max(0)
        self.rng = np.where(self.hi - self.lo > 0, self.hi - self.lo, 1.0); return self
    def t(self, a): return 2 * (np.asarray(a, float) - self.lo) / self.rng - 1
    def inv(self, z): return (np.asarray(z, float) + 1) / 2 * self.rng + self.lo

ACT = {'tansig': (np.tanh, lambda h: 1 - h ** 2),
       'logsig': (lambda z: 1 / (1 + np.exp(-z)), lambda h: h * (1 - h))}

def layers(n, H):
    H = (int(H),) if np.isscalar(H) else tuple(int(h) for h in H)
    return [n, *H, 1]
def ndim(n, H):
    L = layers(n, H); return sum(L[i] * L[i + 1] + L[i + 1] for i in range(len(L) - 1))
def unpack(theta, n, H):
    """theta (..., D) -> [(W (..., out, in), b (..., out)) per layer]; order: W1, b1, W2, b2, ..., W_out, b_out"""
    L = layers(n, H); out = []; i = 0
    for a, b in zip(L[:-1], L[1:]):
        W = theta[..., i:i + a * b].reshape(theta.shape[:-1] + (b, a)); i += a * b
        out.append((W, theta[..., i:i + b])); i += b
    return out

def forward_pop(P, X, n, H, act):
    Ls = unpack(P, n, H); f = ACT[act][0]
    W, b = Ls[0]; A = np.einsum('pon,sn->pso', W, X) + b[:, None, :]
    if len(Ls) > 1: A = f(A)
    for k, (W, b) in enumerate(Ls[1:], 1):
        A = np.einsum('poi,psi->pso', W, A) + b[:, None, :]
        if k < len(Ls) - 1: A = f(A)
    return A[..., 0]

def forward(theta, X, n, H, act):
    return forward_pop(theta[None], X, n, H, act)[0]

def mse_pop(P, X, y, n, H, act):
    return ((forward_pop(P, X, n, H, act) - y[None]) ** 2).mean(1)

# ---------------------------------------------------------------- Levenberg-Marquardt
def jac(theta, X, n, H, act):
    """Jacobian of the network output with respect to theta (S x D) by back-propagation, and the output"""
    Ls = unpack(theta, n, H); f, df = ACT[act]
    acts = [X]; A = X
    for W, b in Ls[:-1]:
        A = f(A @ W.T + b); acts.append(A)
    Wo, bo = Ls[-1]; p = A @ Wo[0] + bo[0]; S = len(X)
    blocks = [np.ones((S, 1)), acts[-1]]                 # reversed order: b_out, W_out
    delta = df(acts[-1]) * Wo[0]                         # dp/dz of the last hidden layer
    for k in range(len(Ls) - 2, -1, -1):
        W, b = Ls[k]
        blocks += [delta, (delta[:, :, None] * acts[k][:, None, :]).reshape(S, -1)]
        if k > 0: delta = (delta @ W) * df(acts[k])
    return np.hstack(blocks[::-1]), p

def lm(theta, X, y, n, H, act, epochs, Xv=None, yv=None, mu=1e-3, hist=None, mon=None):
    """LM for a fixed number of epochs; if (Xv, yv) is given, the weights with the lowest validation MSE are returned.
    hist (list): per-epoch MSE on (X, y), (Xv, yv) and the monitoring set mon = (Xm, ym), in the normalized space."""
    theta = theta.copy(); best = theta.copy()
    def rec(t):
        if hist is not None:
            hist.append((((y - forward(t, X, n, H, act)) ** 2).mean(), ((yv - forward(t, Xv, n, H, act)) ** 2).mean() if Xv is not None else np.nan,
                         ((mon[1] - forward(t, mon[0], n, H, act)) ** 2).mean() if mon is not None else np.nan))
    rec(theta)
    best_v = ((yv - forward(theta, Xv, n, H, act)) ** 2).mean() if Xv is not None else np.inf
    for ep in range(epochs):
        J, p = jac(theta, X, n, H, act); e = y - p; sse = e @ e
        JtJ = J.T @ J; g = J.T @ e
        while True:
            try:
                step = np.linalg.solve(JtJ + mu * np.eye(len(theta)), g)
                t2 = theta + step; e2 = y - forward(t2, X, n, H, act)
                if e2 @ e2 < sse: theta = t2; mu = max(mu * 0.1, 1e-10); break
            except np.linalg.LinAlgError:
                pass
            mu *= 10
            if mu > 1e10: return best if Xv is not None else theta   # converged
        rec(theta)
        if Xv is not None:
            v = ((yv - forward(theta, Xv, n, H, act)) ** 2).mean()
            if v < best_v: best_v, best = v, theta.copy()
    return best if Xv is not None else theta

# ---------------------------------------------------------------- metaheuristics (minimize f(pop) -> fitness vector)
def aso(f, D, lb, ub, N, T, rng, alpha=50.0, beta=0.2):
    """Atom Search Optimization (Zhao et al., 2019), following the authors' reference implementation."""
    X = lb + (ub - lb) * rng.random((N, D)); V = lb + (ub - lb) * rng.random((N, D))
    fit = f(X); b = fit.argmin(); xb, fb = X[b].copy(), fit[b]
    def acc(X, fit, it):
        M = np.exp(-(fit - fit.max()) / (fit.max() - fit.min() + 1e-300)); M = M / M.sum()
        G = np.exp(-20 * it / T)
        K = int(np.floor(N - (N - 2) * (it / T) ** 0.5)) + 1; K = min(K, N)
        idx = np.argsort(-M)[:K]; Xk = X[idx]
        s = np.linalg.norm(X - Xk.mean(0), axis=1)[:, None] + 1e-300          # N x 1
        diff = Xk[None, :, :] - X[:, None, :]                                    # N x K x D
        r = np.linalg.norm(diff, axis=2) + 1e-300                                # N x K
        c = (1 - (it - 1) / T) ** 3
        rmin = 1.1 + 0.1 * np.sin(it / T * np.pi / 2); rs = np.clip(r / s, rmin, 1.24)
        pot = c * (-12 * rs ** -13 + 6 * rs ** -7)                               # 12(-rs)^-13 - 6(-rs)^-7
        E = (rng.random((N, K, D)) * (pot / r)[:, :, None] * diff).sum(1)
        E = alpha * E + beta * (xb - X)
        return E / M[:, None] * G
    A = acc(X, fit, 1)
    for it in range(2, T + 1):
        V = rng.random((N, D)) * V + A
        X = X + V
        bad = (X > ub) | (X < lb); X[bad] = (lb + (ub - lb) * rng.random((N, D)))[bad]
        fit = f(X); b = fit.argmin()
        if fit[b] < fb: xb, fb = X[b].copy(), fit[b]
        A = acc(X, fit, it)
    return xb, fb

def aeo(f, D, lb, ub, N, T, rng, a1=2.0, a2=1.0, GP=0.5, V=1.0):
    """Adaptive Equilibrium Optimizer (Wunnava et al., 2020): EO (Faramarzi et al., 2020) + adaptive dispersal."""
    C = lb + (ub - lb) * rng.random((N, D)); fit = f(C)
    Cold, fold = C.copy(), fit.copy()
    for it in range(1, T + 1):
        better = fit < fold                                         # memory saving
        Cold = np.where(better[:, None], C, Cold); fold = np.where(better, fit, fold)
        C, fit = Cold.copy(), fold.copy()
        order = np.argsort(fit); pool = np.vstack([C[order[:4]], C[order[:4]].mean(0)])
        t = (1 - it / T) ** (a2 * it / T)
        Ceq = pool[rng.integers(0, 5, N)]
        lam = rng.random((N, D)); r = rng.random((N, D))
        Fv = a1 * np.sign(r - 0.5) * (np.exp(-lam * t) - 1)
        r1 = rng.random(N); r2 = rng.random(N)
        GCP = np.where(r2 >= GP, 0.5 * r1, 0.0)[:, None]
        G = GCP * (Ceq - lam * C) * Fv
        C = Ceq + (C - Ceq) * Fv + (G / (lam * V)) * (1 - Fv)
        C = np.clip(C, lb, ub)
        fit = f(C)
        worse = fit >= fit.mean()                                   # adaptive dispersal
        if worse.any():
            C[worse] = np.clip(C[worse] * (0.5 + rng.random((worse.sum(), D))), lb, ub)
            fit[worse] = f(C[worse])
    better = fit < fold
    Cold = np.where(better[:, None], C, Cold); fold = np.where(better, fit, fold)
    b = fold.argmin(); return Cold[b].copy(), fold[b]

def wdo(f, D, lb, ub, N, T, rng, alp=0.4, g=0.2, RT=3.0, c=0.4, maxV=0.3):
    """Wind-Driven Optimization (Bayraktar et al., 2013); positions in [-1, 1] mapped to [lb, ub]."""
    sc = lambda P: lb + (P + 1) / 2 * (ub - lb)
    P = 2 * (rng.random((N, D)) - 0.5); Vv = maxV * 2 * (rng.random((N, D)) - 0.5)
    fit = f(sc(P)); o = np.argsort(fit); P, Vv, fit = P[o], Vv[o], fit[o]
    gp, gf = P[0].copy(), fit[0]
    rank = np.arange(1, N + 1)[:, None]
    for it in range(T):
        perm = np.array([rng.permutation(D) for _ in range(N)])
        other = np.take_along_axis(Vv, perm, 1)
        Vv = (1 - alp) * Vv - g * P + np.abs(1 - 1 / rank) * (gp - P) * RT + c * other / rank
        Vv = np.clip(Vv, -maxV, maxV); P = np.clip(P + Vv, -1, 1)
        fit = f(sc(P)); o = np.argsort(fit); P, Vv, fit = P[o], Vv[o], fit[o]
        if fit[0] < gf: gp, gf = P[0].copy(), fit[0]
    return sc(gp[None])[0], gf

META = {'ASO': aso, 'AEO': aeo, 'WDO': wdo}

# ---------------------------------------------------------------- one model
class Model:
    def __init__(self, method, H, act, seed, N=50, T=500, wlim=2.0, lm_epochs=300, record=False):
        self.method, self.H, self.act, self.seed = method, (int(H),) if np.isscalar(H) else tuple(H), act, seed
        self.N, self.T, self.wlim, self.lm_epochs, self.record = N, T, wlim, lm_epochs, record
    def fit(self, X, y, groups, monitor=None):
        """monitor = (X, y) of an external set whose error is only recorded (learning curves), never used for fitting"""
        rng = np.random.default_rng(self.seed); t0 = time.time()
        self.sx = MinMax().fit(X); self.sy = MinMax().fit(y[:, None])
        Xn = self.sx.t(X); yn = self.sy.t(y[:, None])[:, 0]; n = X.shape[1]; H = self.H; D = ndim(n, H)
        self.n = n; self.nfe = 0; self.meta_hist = []; self.lm_hist = [] if self.record else None
        g = np.unique(groups); hold = rng.choice(g, max(1, int(round(0.2 * len(g)))), replace=False)
        iv = np.isin(groups, hold); Xt, yt, Xv, yv = Xn[~iv], yn[~iv], Xn[iv], yn[iv]
        mon = (self.sx.t(monitor[0]), self.sy.t(monitor[1][:, None])[:, 0]) if (self.record and monitor is not None) else None
        if self.method == 'ANN':
            theta0 = rng.uniform(-0.5, 0.5, D)
        else:
            best = [np.inf]
            def f(P):
                self.nfe += len(P); v = 0.5 * (mse_pop(P, Xt, yt, n, H, self.act) + mse_pop(P, Xv, yv, n, H, self.act))
                if self.record:
                    best[0] = min(best[0], v.min()); self.meta_hist.append((self.nfe, best[0]))
                return v
            theta0, _ = META[self.method](f, D, -self.wlim, self.wlim, self.N, self.T, rng)
        self.theta = lm(theta0, Xt, yt, n, H, self.act, self.lm_epochs, Xv, yv, hist=self.lm_hist, mon=mon)
        self.time = time.time() - t0; self.scale = self.sy.rng[0] / 2     # normalized RMSE x scale = RMSE in MPa
        return self
    def predict(self, X):
        return self.sy.inv(forward(self.theta, self.sx.t(X), self.n, self.H, self.act)[:, None])[:, 0]

class Ensemble:
    """mean of independently trained networks"""
    def __init__(self, members): self.members = members
    def predict_all(self, X): return np.array([m.predict(X) for m in self.members])
    def predict(self, X): return self.predict_all(X).mean(0)
    def predict_sd(self, X):
        P = self.predict_all(X); return P.mean(0), P.std(0, ddof=1)
