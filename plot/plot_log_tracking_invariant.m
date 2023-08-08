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
colors = {'y', 'r', 'g', 'b', 'k'};
for ii = 1:size(data, 1)
    c = colors{data.iter(ii) + 1};
    subplot(2, 1, 1)
    hold on
    plot([0, 1], [qC0{ii, 4}, qCF{ii, 4}], c)
    hold off

    subplot(2, 1, 2)
    hold on
    plot([0, 1], [qdC0{ii, 4}, qdCF{ii, 4}], c)
    hold off
end

figure();
hold on
colors = {'y', 'r', 'g', 'b', 'k'};
for ii = 1:size(data, 1)
    c = colors{data.iter(ii) + 1};
    plot([qC0{ii, 4}, qCF{ii, 4}], [qdC0{ii, 4}, qdCF{ii, 4}], c)
    
end
hold off

%% Bounding Evolution of HLIP States

% Iteration 0
p0_min0 = min(data.cx0(data.iter == 0));
p0_max0 = max(data.cx0(data.iter == 0));
v0_min0 = min(data.cxd0(data.iter == 0));
v0_max0 = max(data.cxd0(data.iter == 0));

p0_minF = min(data.cxF(data.iter == 0));
p0_maxF = max(data.cxF(data.iter == 0));
v0_minF = min(data.cxdF(data.iter == 0));
v0_maxF = max(data.cxdF(data.iter == 0));

fprintf("\np0 in [%0.4f, %0.4f]\nv0 in [%0.4f, %0.4f]\n", p0_min0, p0_max0, v0_min0, v0_max0)
fprintf("p0F in [%0.4f, %0.4f]\nv0F in [%0.4f, %0.4f]\n", p0_minF, p0_maxF, v0_minF, v0_maxF)

p1_min0 = min(data.cx0(data.iter == 1));
p1_max0 = max(data.cx0(data.iter == 1));
v1_min0 = min(data.cxd0(data.iter == 1));
v1_max0 = max(data.cxd0(data.iter == 1));

p1_minF = min(data.cxF(data.iter == 1));
p1_maxF = max(data.cxF(data.iter == 1));
v1_minF = min(data.cxdF(data.iter == 1));
v1_maxF = max(data.cxdF(data.iter == 1));

fprintf("\np1 in [%0.4f, %0.4f]\nv1 in [%0.4f, %0.4f]\n", p1_min0, p1_max0, v1_min0, v1_max0)
fprintf("p1F in [%0.4f, %0.4f]\nv1F in [%0.4f, %0.4f]\n", p1_minF, p1_maxF, v1_minF, v1_maxF)

p2_min0 = min(data.cx0(data.iter == 2));
p2_max0 = max(data.cx0(data.iter == 2));
v2_min0 = min(data.cxd0(data.iter == 2));
v2_max0 = max(data.cxd0(data.iter == 2));

p2_minF = min(data.cxF(data.iter == 2));
p2_maxF = max(data.cxF(data.iter == 2));
v2_minF = min(data.cxdF(data.iter == 2));
v2_maxF = max(data.cxdF(data.iter == 2));

fprintf("\np2 in [%0.4f, %0.4f]\nv2 in [%0.4f, %0.4f]\n", p2_min0, p2_max0, v2_min0, v2_max0)
fprintf("p2F in [%0.4f, %0.4f]\nv2F in [%0.4f, %0.4f]\n", p2_minF, p2_maxF, v2_minF, v2_maxF)