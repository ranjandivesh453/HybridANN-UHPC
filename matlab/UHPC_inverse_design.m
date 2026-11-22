function best = UHPC_inverse_design(target)
%UHPC_INVERSE_DESIGN  Lowest-carbon in-domain mixture whose lower 95% bound
%   at 28 days (28 degC curing) is >= target (MPa). Mirrors the procedure in
%   Section 4.9 of the manuscript (binder 740 kg/m3; convex hull of tested
%   (GGBS, SF) pairs; fibres only with G40/SF12; nearest-neighbour check) on a
%   design grid. The manuscript (Table 16) and the web tool query the NSGA-III
%   Pareto set instead (code/optimize_mix.py), which is finer than this grid,
%   so the two can differ marginally.
S = load(fullfile(fileparts(mfilename('fullpath')), 'UHPC_models.mat'));
d = S.domain;  rho = struct('W',1,'C',3.15,'GGBS',2.91,'SF',2.20,'CA',2.82,'FA',2.51,'SP',1.077,'StF',7.8,'PPF',0.91);
ef = d.EF;   % [C SF GGBS FA CA SP StF PPF W] kg CO2/kg
uc = d.cost; % same order, USD/kg
cand = [];
for SF = 0:3.7:133.2
  for GG = 0:7.4:370
    C = 740 - SF - GG;  if C < 281.2 - 1e-9, continue; end
    opts = [0 0];                                            % [StF PPF]
    if abs(GG-296) < 1e-6 && abs(SF-88.8) < 1e-6
      opts = [opts; [(39:3.9:78)' zeros(11,1)]; [zeros(17,1) (1.82:0.455:9.1)']];
    end
    for k = 1:size(opts,1)
      StF = opts(k,1); PPF = opts(k,2); W = 158 - 10*(StF>0 || PPF>0);
      Vb = C/rho.C + GG/rho.GGBS + SF/rho.SF + W + 8.88/rho.SP + StF/rho.StF + PPF/rho.PPF;
      CA = (d.V_total - Vb) / (1/rho.CA + d.FA_over_CA/rho.FA);  FA = d.FA_over_CA*CA;
      cand(end+1,:) = [W SF GG 28 28 StF PPF C CA FA]; %#ok<AGROW>
    end
  end
end
X = cand(:,[1 8 2 3 4 5 6 7]);   % [W C SF GGBS T Age StF PPF]
inHull = inpolygon(cand(:,3), cand(:,2), d.hull_GGBS_SF(:,1), d.hull_GGBS_SF(:,2));
fib = cand(:,6) > 0 | cand(:,7) > 0;  inHull(fib) = true;   % fibre candidates restricted above
Xn = (X - d.train_min) ./ (d.train_max - d.train_min);
nn = min(pdist2(Xn, d.train_norm), [], 2);
[cs, lo, hi] = UHPC_predict(X);   % selected ten-network ensemble (normalized 95% interval)
Q   = [cand(:,8) cand(:,2) cand(:,3) cand(:,10) cand(:,9) repmat(8.88,size(cand,1),1) cand(:,6) cand(:,7) cand(:,1)];
co2 = Q * ef';  cost = Q * uc';
ok = inHull & nn <= d.nn_threshold & lo >= target;
if ~any(ok), best = []; disp('No in-domain mixture meets the target.'); return; end
idx = find(ok);  [~, j] = sortrows([co2(idx) -lo(idx)]);  i = idx(j(1));
best = struct('Cement',cand(i,8),'GGBS',cand(i,3),'SF',cand(i,2),'Water',cand(i,1),'StF',cand(i,6),'PPF',cand(i,7), ...
              'CA',cand(i,9),'FA',cand(i,10),'Predicted',cs(i),'Lower95',lo(i),'Upper95',hi(i),'CO2',co2(i),'Cost',cost(i),'NNdistance',nn(i));
end
