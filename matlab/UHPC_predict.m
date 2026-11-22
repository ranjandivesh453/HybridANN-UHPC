function [cs, lo, hi, sd] = UHPC_predict(X, modelName)
%UHPC_PREDICT  Compressive strength (MPa) with normalized split-conformal 95% interval.
%   X: n x 8 matrix of inputs in the order [W C SF GGBS T Age StF PPF]
%      (kg/m3; C = 740 - SF - GGBS; T in degC; Age in days).
%   modelName: 'ANN', 'ANN_ASO', 'ANN_AEO' or 'ANN_WDO'; default = selected model (best_model in the .mat file).
%   Each model is an ensemble of ten networks (one or two hidden layers); cs is the member mean and sd the
%   member standard deviation.
%   Interval: cs +/- q95_norm * (sd + gamma). Weights were trained in Python (see ../README.md).
S  = load(fullfile(fileparts(mfilename('fullpath')), 'UHPC_models.mat'));
if nargin < 2, modelName = strtrim(S.best_model); end
e  = S.models.(modelName);
mem = e.members; if ~iscell(mem), mem = num2cell(mem); end
P  = zeros(size(X, 1), numel(mem));
for k = 1:numel(mem)
    m  = mem{k};
    xn = 2 * (X - m.xmin) ./ (m.xmax - m.xmin) - 1;          % mapminmax to [-1, 1]
    W  = m.W; B = m.b; if ~iscell(W), W = num2cell(W); B = num2cell(B); end
    a  = xn;
    for l = 1:numel(W)                                        % hidden layers, then linear output
        a = a * W{l}' + B{l}(:)';
        if l < numel(W)
            if strcmp(strtrim(e.activation), 'tansig'), a = tanh(a); else, a = 1 ./ (1 + exp(-a)); end
        end
    end
    P(:, k) = (a + 1) / 2 * (m.ymax - m.ymin) + m.ymin;
end
cs = mean(P, 2); sd = std(P, 0, 2);
lo = cs - e.q95_norm * (sd + e.gamma);  hi = cs + e.q95_norm * (sd + e.gamma);
end
