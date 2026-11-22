/* HybridANN-UHPC: client-side inference, mixture composition, embodied CO2, cost and domain checks.
 * Mirrors code/uhpc_ann.py (forward pass) and matlab/UHPC_predict.m. Needs model.js (UHPC_MODEL).
 * Inputs x = [W, C, SF, GGBS, T, Age, StF, PPF] in L/m3, kg/m3, kg/m3, kg/m3, degC, days, kg/m3, kg/m3 (C = 740 - SF - GGBS).
 */
(function (root) {
  'use strict';
  const M = () => root.UHPC_MODEL;
  const EPS = 1e-6;

  /** Forward pass of one network (one or more hidden layers, linear output) with [-1, 1] min-max scaling. */
  function forward(n, act, x) {
    let a = x.map((v, i) => 2 * (v - n.xmin[i]) / (n.xmax[i] - n.xmin[i]) - 1);
    const L = n.layers;
    for (let l = 0; l < L.length; l++) {
      const W = L[l].W, b = L[l].b, out = new Array(b.length);
      for (let j = 0; j < b.length; j++) {
        let z = b[j];
        for (let i = 0; i < a.length; i++) z += W[j][i] * a[i];
        out[j] = l < L.length - 1 ? (act === 'tansig' ? Math.tanh(z) : 1 / (1 + Math.exp(-z))) : z;
      }
      a = out;
    }
    return (a[0] + 1) / 2 * (n.ymax - n.ymin) + n.ymin;
  }

  /** Ensemble prediction: mean and standard deviation of the ten member networks. */
  function predictNet(key, x) {
    const e = M().networks[key], v = e.members.map((n) => forward(n, e.activation, x));
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / (v.length - 1));
    return { mean: m, sd: sd };
  }

  /** Prediction and normalized split-conformal 95% interval for every model. */
  function predictAll(x) {
    const out = {};
    for (const key of Object.keys(M().networks)) {
      const e = M().networks[key], p = predictNet(key, x), h = e.q95_norm * (p.sd + e.gamma);
      out[key] = { label: e.label, mean: p.mean, sd: p.sd, lo: p.mean - h, hi: p.mean + h, half: h };
    }
    return out;
  }

  /** Full mixture (kg/m3) from the model inputs; aggregates by the absolute-volume equation. */
  function composition(inp) {
    const d = M().domain, r = M().materials.density;
    const C = d.binder - inp.SF - inp.GGBS;
    const Vb = C / r.C + inp.GGBS / r.GGBS + inp.SF / r.SF + inp.W / r.W + d.sp / r.SP + inp.StF / r.StF + inp.PPF / r.PPF;
    const CA = (d.V_total - Vb) / (1 / r.CA + d.FA_over_CA / r.FA);
    return { C: C, GGBS: inp.GGBS, SF: inp.SF, W: inp.W, SP: d.sp, CA: CA, FA: d.FA_over_CA * CA, StF: inp.StF, PPF: inp.PPF };
  }
  const sumWith = (mix, f) => Object.keys(f).reduce((s, k) => s + (mix[k] || 0) * f[k], 0);
  const co2 = (mix) => sumWith(mix, M().materials.co2);    // kg CO2/m3 (A1-A3)
  const cost = (mix) => sumWith(mix, M().materials.cost);  // USD/m3

  /** Minimum Euclidean distance to the training records in [0, 1]-normalized input space. */
  function nnDistance(x) {
    const d = M().domain, lo = d.train_min, hi = d.train_max;
    const z = x.map((v, i) => (v - lo[i]) / (hi[i] - lo[i]));
    let best = Infinity;
    for (const t of d.train_norm) {
      let s = 0;
      for (let i = 0; i < z.length; i++) { const e = z[i] - t[i]; s += e * e; }
      if (s < best) best = s;
    }
    return Math.sqrt(best);
  }

  /** Point-in-convex-polygon test for (GGBS, SF) against the hull of tested plain compositions. */
  function inHull(ggbs, sf) {
    const v = M().domain.hull_GGBS_SF; let sign = 0;
    for (let i = 0; i < v.length; i++) {
      const a = v[i], b = v[(i + 1) % v.length];
      const c = (b[0] - a[0]) * (sf - a[1]) - (b[1] - a[1]) * (ggbs - a[0]);
      if (Math.abs(c) < 1e-6) continue;
      const s = Math.sign(c);
      if (sign === 0) sign = s; else if (s !== sign) return false;
    }
    return true;
  }

  const near = (a, b, tol) => Math.abs(a - b) <= (tol || 0.05);

  /** Applicability-domain checks; returns {inside, warnings[], nn}. */
  function domainCheck(x) {
    const [W, C, SF, GGBS, T, Age, StF, PPF] = x, d = M().domain, names = M().inputs, w = [];
    names.forEach((nm, i) => {
      if (x[i] < d.train_min[i] - EPS || x[i] > d.train_max[i] + EPS)
        w.push(`${nm} = ${fmt(x[i])} is outside the training range ${fmt(d.train_min[i])}–${fmt(d.train_max[i])}.`);
    });
    if (Math.abs(C + SF + GGBS - d.binder) > 0.5) w.push(`OPC + SF + GGBS = ${fmt(C + SF + GGBS)} kg/m³; all tested mixtures had a binder content of ${d.binder} kg/m³.`);
    if (C < d.cement_min - EPS) w.push(`Cement = ${fmt(C)} kg/m³ is below the lowest tested content (${fmt(d.cement_min)} kg/m³).`);
    const fib = StF > 0 || PPF > 0;
    if (StF > 0 && PPF > 0) w.push('No tested mixture contained both steel and PP fibres.');
    if (!fib) {
      if (!inHull(GGBS, SF)) w.push('The (GGBS, SF) combination lies outside the convex hull of the tested plain binders.');
      if (!near(W, d.water_plain)) w.push(`Plain mixtures were batched with ${d.water_plain} L/m³ of water.`);
    } else {
      const g40sf12 = near(SF, d.fibre_binder.SF) && near(GGBS, d.fibre_binder.GGBS);
      const heat = StF > 0 && near(StF, 78) && near(T, 120) && near(SF, 88.8) && [74, 148, 222, 296, 370].some((g) => near(GGBS, g));
      if (!g40sf12 && !heat) w.push('Fibres were tested only with the G40/SF12 binder (and 1.0% steel fibre with G10–G50/SF12 at 120 °C).');
      if (!near(T, 28) && !heat) w.push('Fibre-reinforced mixtures were cured at 28 °C (1.0% steel fibre also at 120 °C).');
      if (StF > 0 && (StF < d.StF_range[0] - EPS || StF > d.StF_range[1] + EPS)) w.push(`Steel fibre was tested at ${d.StF_range[0]}–${d.StF_range[1]} kg/m³ (0.5–1.0 vol.%).`);
      if (PPF > 0 && (PPF < d.PPF_range[0] - EPS || PPF > d.PPF_range[1] + EPS)) w.push(`PP fibre was tested at ${d.PPF_range[0]}–${d.PPF_range[1]} kg/m³ (0.2–1.0 vol.%).`);
      if (!near(W, d.water_fibre)) w.push(`Fibre mixtures were batched with ${d.water_fibre} L/m³ of water.`);
    }
    const nn = nnDistance(x);
    if (nn > d.nn_threshold + EPS) w.push(`Nearest-neighbour distance to the training data is ${nn.toFixed(3)} (limit ${d.nn_threshold.toFixed(2)}).`);
    return { inside: w.length === 0, warnings: w, nn: nn };
  }

  /** Measured strengths in the database for exactly these inputs. */
  function measured(x) {
    const rec = M().records.filter((r) => x.every((v, i) => Math.abs(v - r[i]) < 1e-6));
    if (!rec.length) return null;
    const cs = rec.map((r) => r[8]);
    return { mean: cs.reduce((a, b) => a + b, 0) / cs.length, n: cs.length, mixture: rec[0][10], subset: rec[0][9] };
  }

  function fmt(v) { return Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : (+v.toFixed(2)).toString(); }

  root.UHPC = { predictNet, predictAll, composition, co2, cost, nnDistance, inHull, domainCheck, measured, fmt };
})(typeof window !== 'undefined' ? window : globalThis);
