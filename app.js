/* HybridANN-UHPC project page: interactive mix designer (Predict / Inverse design / Optimize mix). */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const M = window.UHPC_MODEL, U = window.UHPC, T = window.UHPC_TOPSIS, P = window.PARETO_SOLUTIONS, META = window.PARETO_META;
  const f1 = (v) => (+v).toFixed(1), f2 = (v) => (+v).toFixed(2), f0 = (v) => Math.round(+v).toString();
  const B = M.domain.binder, RHO = M.materials.density;

  /* ---------------- tabs ---------------- */
  document.querySelectorAll('.tab-btn').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
  function showTab(id) {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === id));
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === id));
    if (id === 'tab-optimize') runTopsis();
  }

  /* ---------------- predict ---------------- */
  const sel = $('preset');
  M.mixtures.slice().sort((a, b) => a.name.localeCompare(b.name, 'en', { numeric: true })).forEach((m) => {
    const o = document.createElement('option'); o.value = m.name; o.textContent = m.name; sel.appendChild(o);
  });
  sel.addEventListener('change', () => {
    const m = M.mixtures.find((x) => x.name === sel.value); if (!m) return;
    $('inSF').value = m.SF; $('inGGBS').value = m.GGBS; $('inW').value = m.W;
    $('inFibre').value = m.StF > 0 ? 'steel' : m.PPF > 0 ? 'pp' : 'none';
    $('inFib').value = m.StF > 0 ? m.StF : m.PPF > 0 ? m.PPF : 0;
    if (m.StF > 0 || m.PPF > 0) { if (!(m.StF === 78 && +$('inT').value === 120)) $('inT').value = 28; }
    syncFibre(false); runPredict();
  });
  $('inFibre').addEventListener('change', () => syncFibre(true));
  function syncFibre(setDefaults) {
    const t = $('inFibre').value, fib = $('inFib');
    fib.disabled = t === 'none';
    if (setDefaults) {
      fib.value = t === 'steel' ? 78 : t === 'pp' ? 1.82 : 0;
      $('inW').value = t === 'none' ? M.domain.water_plain : M.domain.water_fibre;
      if (t !== 'none') { $('inSF').value = 88.8; $('inGGBS').value = 296; $('inT').value = 28; }
    }
    $('fibUnit').textContent = t === 'none' ? '' : '≈ ' + f2(+fib.value / (t === 'steel' ? RHO.StF : RHO.PPF) / 10) + ' vol.%';
    runPredict();
  }
  ['inSF', 'inGGBS', 'inT', 'inAge', 'inFib', 'inW', 'inModel'].forEach((id) => $(id).addEventListener('input', () => {
    if (id === 'inFib') syncFibre(false); else runPredict();
  }));
  $('predictBtn').addEventListener('click', runPredict);

  function inputs() {
    const t = $('inFibre').value, v = +$('inFib').value || 0;
    const sf = +$('inSF').value, gg = +$('inGGBS').value;
    return [+$('inW').value, B - sf - gg, sf, gg, +$('inT').value, +$('inAge').value, t === 'steel' ? v : 0, t === 'pp' ? v : 0];
  }
  function runPredict() {
    const x = inputs(), key = $('inModel').value;
    $('cementOut').textContent = f1(x[1]) + ' kg/m³ (' + f1(100 * x[1] / B) + '% of binder)';
    if (x.some((v) => !isFinite(v))) { $('result').innerHTML = '<p class="warn">Please enter numeric values.</p>'; return; }
    const all = U.predictAll(x), p = all[key], dc = U.domainCheck(x), meas = U.measured(x);
    const mix = U.composition({ W: x[0], SF: x[2], GGBS: x[3], StF: x[6], PPF: x[7] });
    const co2 = U.co2(mix), cost = U.cost(mix);
    const status = dc.inside
      ? '<span class="badge ok">Inside the experimental domain</span>'
      : '<span class="badge warn">Outside the experimental domain – extrapolation</span>';
    const rows = Object.keys(all).map((k) => `<tr${k === key ? ' class="sel"' : ''}><td>${all[k].label}${k === M.selected ? ' <span class="tag">selected model</span>' : ''}</td><td>${f2(all[k].mean)}</td><td>${f2(all[k].lo)} – ${f2(all[k].hi)}</td></tr>`).join('');
    const vf = x[6] / RHO.StF / 10 + x[7] / RHO.PPF / 10;
    $('result').innerHTML = `
      <div class="res-grid">
        <div class="res-main">
          <div class="res-label">Predicted compressive strength (${p.label})</div>
          <div class="res-value">${f1(p.mean)} <span>MPa</span></div>
          <div class="res-pi">95% prediction interval: <b>${f1(p.lo)} – ${f1(p.hi)} MPa</b> (normalized split-conformal, ±${f2(p.half)} MPa; ensemble SD ${f2(p.sd)} MPa)</div>
          ${meas ? `<div class="res-meas">Measured in the database: <b>${f2(meas.mean)} MPa</b> (${meas.mixture}, ${meas.n} record${meas.n > 1 ? 's' : ''}, ${meas.subset.toLowerCase()} set)</div>` : ''}
          <div style="margin-top:.6rem">${status}</div>
          ${dc.warnings.length ? '<ul class="warn-list">' + dc.warnings.map((w) => '<li>' + w + '</li>').join('') + '</ul>' : ''}
        </div>
        <div class="res-side">
          <table class="kv">
            <tr><th colspan="2">Mixture (1 m³)</th></tr>
            <tr><td>OPC / GGBS / SF</td><td>${f1(mix.C)} / ${f1(mix.GGBS)} / ${f1(mix.SF)} kg</td></tr>
            <tr><td>w/b · SCM share</td><td>${(x[0] / B).toFixed(3)} · ${f1(100 * (x[2] + x[3]) / B)}%</td></tr>
            <tr><td>Coarse / fine aggregate</td><td>${f0(mix.CA)} / ${f0(mix.FA)} kg</td></tr>
            <tr><td>Fibre volume</td><td>${f2(vf)}%</td></tr>
            <tr><th colspan="2">Sustainability (A1–A3)</th></tr>
            <tr><td>Embodied CO₂</td><td><b>${f0(co2)} kg/m³</b></td></tr>
            <tr><td>Material cost</td><td><b>${f0(cost)} USD/m³</b></td></tr>
            <tr><td>CO₂ per MPa · cost per MPa</td><td>${f2(co2 / p.mean)} · ${f2(cost / p.mean)}</td></tr>
          </table>
        </div>
      </div>
      <table class="results-table compact"><thead><tr><th>Network</th><th>Prediction (MPa)</th><th>95% interval (MPa)</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="note">CO₂ and cost use the factors in <code>model.json</code> (Alsomiri et al. 2026; Wakjira &amp; Goshu 2026). Heat-curing energy is not included.</p>`;
  }

  /* ---------------- inverse design ---------------- */
  const tg = $('targetStrength'), tgn = $('targetStrengthNum');
  tg.addEventListener('input', () => { tgn.value = tg.value; runInverse(); });
  tgn.addEventListener('input', () => { tg.value = tgn.value; runInverse(); });
  function mixCard(title, s, extra) {
    const fib = s.steel_fiber > 0 ? `Steel fibre ${f1(s.steel_fiber)} kg/m³ (${f2(s.vf)} vol.%)` : s.pp_fiber > 0 ? `PP fibre ${f2(s.pp_fiber)} kg/m³` : 'No fibres';
    return `<div class="mix-card">
      <div class="mix-title">${title}</div>
      <div class="mix-strength">${f1(s.strength)} <span>MPa</span></div>
      <div class="mix-pi">95% PI ${f1(s.lower95)} – ${f1(s.upper95)} MPa</div>
      <table class="kv">
        <tr><td>OPC</td><td>${f1(s.cement)} kg/m³ (${f1(100 * s.cement / B)}%)</td></tr>
        <tr><td>GGBS</td><td>${f1(s.ggbs)} kg/m³ (${f1(100 * s.ggbs / B)}%)</td></tr>
        <tr><td>Silica fume</td><td>${f1(s.silica_fume)} kg/m³ (${f1(100 * s.silica_fume / B)}%)</td></tr>
        <tr><td>Water · SP</td><td>${f0(s.water)} L/m³ · ${f2(s.superplasticizer)} kg/m³</td></tr>
        <tr><td>Coarse / fine agg.</td><td>${f0(s.coarse_agg)} / ${f0(s.fine_agg)} kg/m³</td></tr>
        <tr><td>Fibres</td><td>${fib}</td></tr>
        <tr><td>Embodied CO₂</td><td><b>${f0(s.co2)} kg/m³</b></td></tr>
        <tr><td>Material cost</td><td><b>${f0(s.cost)} USD/m³</b></td></tr>
        ${extra || ''}
      </table>
      <button class="btn-small" data-load="${s.id}">Open in predictor</button>
    </div>`;
  }
  function bindLoad(el) {
    el.querySelectorAll('[data-load]').forEach((b) => b.addEventListener('click', () => {
      const s = P.find((x) => x.id === +b.dataset.load); loadSolution(s); showTab('tab-predict');
      $('predictor').scrollIntoView({ behavior: 'smooth' });
    }));
  }
  function loadSolution(s) {
    $('inSF').value = s.silica_fume; $('inGGBS').value = s.ggbs; $('inW').value = s.water; $('inT').value = META.curing_temp; $('inAge').value = META.age;
    $('inFibre').value = s.steel_fiber > 0 ? 'steel' : s.pp_fiber > 0 ? 'pp' : 'none';
    $('inFib').value = s.steel_fiber > 0 ? s.steel_fiber : s.pp_fiber; $('inModel').value = M.selected; sel.value = ''; syncFibre(false);
  }
  function runInverse() {
    const t = +tg.value, a = T.findInverse(P, t, 'co2'), b = T.findInverse(P, t, 'cost'), el = $('inverseResult');
    if (!a) {
      const mx = Math.max(...P.map((s) => s.lower95));
      el.innerHTML = `<p class="warn">No mixture inside the experimental domain reaches a lower 95% bound of ${t} MPa at 28 days. The highest attainable lower bound is ${f1(mx)} MPa (G40/SF12 with 1.0 vol.% steel fibre).</p>`;
      return;
    }
    el.innerHTML = `<div class="mix-grid two">${mixCard('Lowest-carbon design', a)}${b.id === a.id ? '' : mixCard('Lowest-cost design', b)}</div>
      ${b.id === a.id ? '<p class="note">The lowest-carbon design is also the lowest-cost design for this target.</p>' : ''}`;
    bindLoad(el);
  }

  /* ---------------- optimize (TOPSIS) ---------------- */
  const W = ['wStr', 'wCO2', 'wCost'];
  W.forEach((id) => $(id).addEventListener('input', runTopsis));
  document.querySelectorAll('[data-w]').forEach((b) => b.addEventListener('click', () => {
    b.dataset.w.split(',').forEach((v, i) => { $(W[i]).value = v; }); runTopsis();
  }));
  $('xAxis').addEventListener('change', runTopsis);
  function runTopsis() {
    const w = W.map((id) => +$(id).value), s = w.reduce((a, b) => a + b, 0) || 1;
    W.forEach((id, i) => { $(id + 'Val').textContent = (w[i] / s).toFixed(2); });
    const ranked = T.topsis(P, w), best = ranked[0];
    const top = ranked.slice(0, 5).map((r, i) => `<tr${i === 0 ? ' class="sel"' : ''}><td>${i + 1}</td><td>${f1(r.cement)} / ${f1(r.ggbs)} / ${f1(r.silica_fume)}</td><td>${r.steel_fiber > 0 ? 'StF ' + f1(r.steel_fiber) : '–'}</td><td>${f1(r.strength)} [${f1(r.lower95)}–${f1(r.upper95)}]</td><td>${f0(r.co2)}</td><td>${f0(r.cost)}</td><td>${r.score.toFixed(3)}</td></tr>`).join('');
    $('topsisResult').innerHTML = `<div class="mix-grid two">${mixCard('TOPSIS best compromise', best, `<tr><td>Closeness score</td><td>${best.score.toFixed(3)}</td></tr>`)}
      <div><div class="chart-head">Pareto front (${P.length} solutions, 28 °C curing, 28 days)</div><div id="chart"></div></div></div>
      <table class="results-table compact"><thead><tr><th>Rank</th><th>OPC / GGBS / SF (kg/m³)</th><th>Fibre</th><th>f<sub>c</sub> [95% PI] (MPa)</th><th>CO₂ (kg/m³)</th><th>Cost (USD/m³)</th><th>Score</th></tr></thead><tbody>${top}</tbody></table>`;
    bindLoad($('topsisResult'));
    drawChart(best);
  }
  const tested = M.mixtures.map((m) => {
    const r = M.records.filter((x) => x[10] === m.name && x[4] === 28 && x[5] === 28);
    if (!r.length) return null;
    const mix = U.composition(m);
    return { name: m.name, co2: U.co2(mix), cost: U.cost(mix), strength: r.reduce((a, b) => a + b[8], 0) / r.length };
  }).filter(Boolean);
  function drawChart(best) {
    const xk = $('xAxis').value, W0 = 460, H0 = 330, m = { l: 48, r: 14, t: 12, b: 40 };
    const all = P.concat(tested), xs = all.map((s) => s[xk]), ys = all.map((s) => s.strength);
    const x0 = Math.floor(Math.min(...xs) / 50) * 50, x1 = Math.ceil(Math.max(...xs) / 50) * 50, y0 = 80, y1 = 160;
    const X = (v) => m.l + (v - x0) / (x1 - x0) * (W0 - m.l - m.r), Y = (v) => H0 - m.b - (v - y0) / (y1 - y0) * (H0 - m.t - m.b);
    const cmin = Math.min(...P.map((s) => s.cost)), cmax = Math.max(...P.map((s) => s.cost));
    const shade = (c) => { const t = (c - cmin) / (cmax - cmin); const l = 82 - 52 * t; return `hsl(214,62%,${l}%)`; };
    let g = '';
    for (let v = x0; v <= x1; v += 50) g += `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${H0 - m.b}" class="grid"/><text x="${X(v)}" y="${H0 - m.b + 14}" class="tick" text-anchor="middle">${v}</text>`;
    for (let v = y0; v <= y1; v += 20) g += `<line x1="${m.l}" x2="${W0 - m.r}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/><text x="${m.l - 6}" y="${Y(v) + 3}" class="tick" text-anchor="end">${v}</text>`;
    const pts = P.map((s) => `<circle cx="${X(s[xk]).toFixed(1)}" cy="${Y(s.strength).toFixed(1)}" r="2.6" fill="${shade(s.cost)}" data-id="${s.id}"/>`).join('');
    const tp = tested.map((s) => `<circle cx="${X(s[xk]).toFixed(1)}" cy="${Y(s.strength).toFixed(1)}" r="3.6" class="tested" data-t="${s.name}"/>`).join('');
    const bx = X(best[xk]), by = Y(best.strength);
    $('chart').innerHTML = `<svg viewBox="0 0 ${W0} ${H0}" class="chart" role="img" aria-label="Pareto front: strength versus ${xk === 'co2' ? 'embodied CO2' : 'cost'}">
      ${g}<line x1="${m.l}" x2="${W0 - m.r}" y1="${Y(120)}" y2="${Y(120)}" class="thr"/>
      <text x="${m.l + 6}" y="${Y(120) + 13}" class="tick" text-anchor="start">lower 95% bound ≥ 120 MPa</text>
      ${tp}${pts}<circle cx="${bx}" cy="${by}" r="7" class="best"/>
      <text x="${(m.l + W0 - m.r) / 2}" y="${H0 - 6}" class="axis" text-anchor="middle">${xk === 'co2' ? 'Embodied CO₂ (kg/m³)' : 'Material cost (USD/m³)'}</text>
      <text transform="translate(12,${(m.t + H0 - m.b) / 2}) rotate(-90)" class="axis" text-anchor="middle">28-day f<tspan baseline-shift="sub" font-size="9">c</tspan> (MPa)</text></svg>
      <div class="legend"><span><i class="dot p"></i>Pareto solution (darker = costlier)</span><span><i class="dot t"></i>Tested mixture (measured)</span><span><i class="dot b"></i>TOPSIS choice</span></div>`;
    const tip = $('tip'), svg = $('chart').querySelector('svg');
    svg.addEventListener('mousemove', (e) => {
      const c = e.target.closest('circle');
      if (!c || c.classList.contains('best')) { tip.style.display = 'none'; return; }
      let html;
      if (c.dataset.t) { const s = tested.find((x) => x.name === c.dataset.t); html = `<b>${s.name}</b> (tested)<br>${f2(s.strength)} MPa measured<br>${f0(s.co2)} kg CO₂/m³ · ${f0(s.cost)} USD/m³`; }
      else { const s = P.find((x) => x.id === +c.dataset.id); html = `<b>Solution #${s.id}</b><br>OPC/GGBS/SF ${f0(s.cement)}/${f0(s.ggbs)}/${f0(s.silica_fume)}${s.steel_fiber > 0 ? ' + StF ' + f0(s.steel_fiber) : ''}<br>${f1(s.strength)} MPa [${f1(s.lower95)}–${f1(s.upper95)}]<br>${f0(s.co2)} kg CO₂/m³ · ${f0(s.cost)} USD/m³`; }
      tip.innerHTML = html; tip.style.display = 'block'; tip.style.left = (e.pageX + 12) + 'px'; tip.style.top = (e.pageY + 12) + 'px';
    });
    svg.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
  }

  /* ---------------- representative solutions table ---------------- */
  (function reps() {
    const names = { 'Maximum strength': 'Maximum strength', 'Minimum CO2': 'Minimum CO₂ (= best efficiency)', 'Minimum cost': 'Minimum cost', 'Balanced (knee)': 'Balanced (knee)', 'TOPSIS (equal weights)': 'TOPSIS, equal weights' };
    const rows = Object.keys(names).map((k) => {
      const s = P.find((x) => x.id === META.representatives[k]);
      return `<tr><td>${names[k]}</td><td>${f1(s.cement)} / ${f1(s.ggbs)} / ${f1(s.silica_fume)}${s.steel_fiber > 0 ? ' + StF ' + f1(s.steel_fiber) : ''}</td><td>${f1(s.strength)} [${f1(s.lower95)}–${f1(s.upper95)}]</td><td>${f0(s.co2)}</td><td>${f0(s.cost)}</td><td><button class="btn-small" data-load="${s.id}">Open</button></td></tr>`;
    }).join('');
    $('repTable').innerHTML = `<thead><tr><th>Solution</th><th>OPC / GGBS / SF (kg/m³)</th><th>f<sub>c,28</sub> [95% PI] (MPa)</th><th>CO₂ (kg/m³)</th><th>Cost (USD/m³)</th><th></th></tr></thead><tbody>${rows}</tbody>`;
    bindLoad($('repTable'));
  })();

  /* ---------------- BibTeX ---------------- */
  $('copy-bibtex').addEventListener('click', () => {
    const t = $('bibtex-text').textContent;
    const done = () => { $('copy-bibtex').textContent = 'Copied'; setTimeout(() => { $('copy-bibtex').textContent = 'Copy'; }, 1500); };
    if (navigator.clipboard) navigator.clipboard.writeText(t).then(done, done); else done();
  });

  /* ---------------- init ---------------- */
  $('nPareto').textContent = P.length; $('nParetoB').textContent = P.length + ' Pareto-optimal';
  syncFibre(false); runInverse();
  if (location.hash === '#inverse') showTab('tab-inverse');
  if (location.hash === '#optimize') showTab('tab-optimize');
})();
