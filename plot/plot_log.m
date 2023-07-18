close all;
clear;
clc;

%% Get log data

% Specify the relative location of the log file
file_name = "log_ctrl.csv";
% Store the log data into a table
data = readtable(file_name);

t = data.t; t_phase = data.tphase; t_scaled = data.tscaled;
x = data.x; z = data.z; pitch = data.pitch; q1 = data.q1; q2 = data.q2; q3 = data.q3; q4 = data.q4;
xdot = data.xdot; zdot = data.zdot; pitchdot = data.pitchdot; q1dot = data.q1dot; q2dot = data.q2dot; q3dot = data.q3dot; q4dot = data.q4dot;
xkin = data.xkin; zkin = data.zkin; pitchkin = data.pitchkin; q1kin = data.q1kin; q2kin = data.q2kin; q3kin = data.q3kin; q4kin = data.q4kin;
vrefgoal = data.vrefgoal; vref = data.vref; zrefgoal = data.zrefgoal; zref = data.zref;
x_curr = data.x_ssp_curr; v_curr = data.v_ssp_curr; x_impact = data.x_ssp_impact; v_impact = data.v_ssp_impact; x_impact_ref = data.x_ssp_impact_ref; v_impact_ref = data.v_ssp_impact_ref;
u_nom = data.unom; u = data.u; bht = data.bht;
pitchref = data.pitchref; x_swf_ref = data.swfx; z_swf_ref = data.swfz; x_com_ref = data.comx; z_com_ref = data.comz;
q1ref = data.q1ref; q2ref = data.q2ref; q3ref = data.q3ref; q4ref = data.q4ref;
%% Plot Joints

fh1 = figure();
subplot(2, 1, 1)
hold on;
plot(t, x)
plot(t, z)
plot(t, pitch)
hold off;
legend('x', 'z', 'pitch')
subplot(2, 1, 2)
hold on;
plot(t, q1)
plot(t, q2)
plot(t, q3)
plot(t, q4)
hold off;
legend('q1', 'q2', 'q3', 'q4')
sgtitle('Generalized Position')

%% Plot Velocities
fh2 = figure();
subplot(2, 1, 1)
hold on;
plot(t, xdot)
plot(t, zdot)
plot(t, pitchdot)
hold off;
legend('xdot', 'zdot', 'pitchdot')
subplot(2, 1, 2)
hold on;
plot(t, q1dot)
plot(t, q2dot)
plot(t, q3dot)
plot(t, q4dot)
hold off;
legend('q1dot', 'q2dot', 'q3dot', 'q4dot')
sgtitle('Generalized Velocity')

%% Plot Joint References
fh3 = figure();
hold on;
plot(t, q1ref)
plot(t, q2ref)
plot(t, q3ref)
plot(t, q4ref)
hold off;
legend('q1ref', 'q2ref', 'q3ref', 'q4ref')
title('Joint Reference')

%% Plot Step Lengths
fh4 = figure();
subplot(2, 1, 1)
hold on;
plot(t, u_nom)
plot(t, u)
hold off
legend('u nominal', 'u')
subplot(2, 1, 2)
hold on
plot(t, x_swf_ref)
plot(t, z_swf_ref)
hold off
legend('x swf ref', 'z swf ref')
sgtitle('Step Reference')

%% Plot HLIP Predictions
fh5 = figure();
subplot(2, 1, 1)
hold on
plot(t, x_curr)
plot(t, x_impact)
plot(t, x_impact_ref)
hold off;
legend('x com', 'predicted x com impact', 'x com impact ref')
subplot(2, 1, 2)
hold on
plot(t, v_curr)
plot(t, v_impact)
plot(t, v_impact_ref)
hold off
legend('v com', 'predicted v com impact', 'v com impact ref')
sgtitle('HLIP Approximation')


%% Plot Tracking Accuracy
figure();
subplot(2, 2, 1)
hold on
plot(t, q1)
plot(t, q1ref)
hold off
legend('q1', 'q1ref')
subplot(2, 2, 2)
hold on
plot(t, q2)
plot(t, q2ref)
hold off
legend('q2', 'q2ref')
subplot(2, 2, 3)
hold on
plot(t, q3)
plot(t, q3ref)
hold off
legend('q3', 'q3ref')
subplot(2, 2, 4)
hold on
plot(t, q4)
plot(t, q4ref)
hold off
legend('q4', 'q4ref')
sgtitle('Tracking performance')
