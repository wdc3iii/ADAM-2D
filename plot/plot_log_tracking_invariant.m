close all;
clear;
clc;

%% Get log data

% Specify the relative location of the log file
file_name= "log_tracking_invariant.csv";
% Store the log data into a table
data = readtable(file_name);
q0 = data(:, 2:8);
qd0 = data(:, 9:15);
qC0 = data(:, 16:20);
qdC0 = data(:, 21:25);
qF = data(:, 26:32);
qdF = data(:, 33:39);
qCF = data(:, 40:44);
qdCF = data(:, 45:49);

%% Evolution of HLIP States
figure();
y = [0.9290 0.6940 0.1250];
p = [0.4940 0.1840 0.5560];
colors = [linspace(y(1), p(1), max(data.iter) + 1); linspace(y(2), p(2), max(data.iter) + 1); linspace(y(3), p(3), max(data.iter) + 1)];
% colors = {'r', 'g', 'b', 'c', 'y', 'm', 'k'};
for ii = 1:size(data, 1)
    c = colors(:, data.iter(ii) + 1);
    subplot(2, 1, 1)
    hold on
    plot([0, 1], [qC0{ii, 4}, qCF{ii, 4}], 'color', c)
    hold off

    subplot(2, 1, 2)
    hold on
    plot([0, 1], [qdC0{ii, 4}, qdCF{ii, 4}], 'color', c)
    hold off
end

figure();
hold on
for ii = 1:size(data, 1)
    c = colors(:, data.iter(ii) + 1);
    plot([qC0{ii, 4}, qCF{ii, 4}], [qdC0{ii, 4}, qdCF{ii, 4}], 'color', c)
    
end
hold off

%% Bounding Evolution of HLIP States
pmax = -inf;
pmin = inf;
vmax = -inf;
vmin = inf;
% Iteration 0
for ii = 1:max(data.iter)
    FI = true;
    p_min0 = min(data.cx0(data.iter == ii));
    p_max0 = max(data.cx0(data.iter == ii));
    v_min0 = min(data.cxd0(data.iter == ii));
    v_max0 = max(data.cxd0(data.iter == ii));

    if p_min0 < pmin
        pmin = p_min0;
    end
    if p_max0 > pmax
        pmax = p_max0;
    end
    if v_min0 < vmin
        vmin = v_min0;
    end
    if v_max0 > vmax
        vmax = v_max0;
    end
    
    p_minF = min(data.cxF(data.iter == ii));
    p_maxF = max(data.cxF(data.iter == ii));
    v_minF = min(data.cxdF(data.iter == ii));
    v_maxF = max(data.cxdF(data.iter == ii));

    if p_minF < pmin
        pmin = p_minF;
        FI = false;
    end
    if p_maxF > pmax
        pmax = p_maxF;
        FI = false;
    end
    if v_minF < vmin
        vmin = v_minF;
        FI = false;
    end
    if v_maxF > vmax
        vmax = v_maxF;
        FI = false;
    end
    
    fprintf("\nIter: %d\np0 in [%0.4f, %0.4f]\nv0 in [%0.4f, %0.4f]\n", ii, p_min0, p_max0, v_min0, v_max0)
    fprintf("p0F in [%0.4f, %0.4f]\nv0F in [%0.4f, %0.4f]\n", p_minF, p_maxF, v_minF, v_maxF)
    fprintf("p0 scope [%0.4f, %0.4f]\nv0 scope [%0.4f, %0.4f]\n", pmin, pmax, vmin, vmax)
    if FI
        disp("Samples forward invariant! (in HLIP projection at least)")
    end
end