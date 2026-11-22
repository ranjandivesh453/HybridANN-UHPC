/* HybridANN-UHPC: TOPSIS ranking and inverse design on the Pareto set (pareto_data.js).
 * Criteria: strength (benefit), co2 (cost), cost (cost). Vector normalization, as in code/optimize_mix.py.
 */
(function (root) {
  'use strict';

  /** Returns solutions with a TOPSIS closeness score, best first. weights = [strength, co2, cost]. */
  function topsis(solutions, weights) {
    const keys = ['strength', 'co2', 'cost'], benefit = [true, false, false];
    const ws = weights.reduce((a, b) => a + b, 0) || 1, w = weights.map((v) => v / ws);
    const norm = keys.map((k) => Math.sqrt(solutions.reduce((s, x) => s + x[k] * x[k], 0)));
    const V = solutions.map((x) => keys.map((k, j) => (x[k] / norm[j]) * w[j]));
    const best = keys.map((_, j) => (benefit[j] ? Math.max : Math.min)(...V.map((v) => v[j])));
    const worst = keys.map((_, j) => (benefit[j] ? Math.min : Math.max)(...V.map((v) => v[j])));
    const dist = (v, ref) => Math.sqrt(v.reduce((s, x, j) => s + (x - ref[j]) ** 2, 0));
    return solutions.map((x, i) => {
      const dp = dist(V[i], best), dn = dist(V[i], worst);
      return Object.assign({}, x, { score: dp + dn > 0 ? dn / (dp + dn) : 0 });
    }).sort((a, b) => b.score - a.score);
  }

  /** Pareto solution whose lower 95% bound meets the target, with the lowest CO2 (by='co2') or cost (by='cost'). */
  function findInverse(solutions, target, by) {
    const a = by === 'cost' ? 'cost' : 'co2', b = a === 'co2' ? 'cost' : 'co2';
    const ok = solutions.filter((s) => s.lower95 >= target);
    if (!ok.length) return null;
    return ok.reduce((m, s) => (s[a] < m[a] - 1e-9 || (Math.abs(s[a] - m[a]) <= 1e-9 && s[b] < m[b]) ? s : m));
  }

  root.UHPC_TOPSIS = { topsis, findInverse };
})(typeof window !== 'undefined' ? window : globalThis);
