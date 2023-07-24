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
y_pitchref = data.ypitchref; x_swf_ref = data.yswfxref; z_swf_ref = data.yswfzref; x_com_ref = data.ycomxref; z_com_ref = data.ycomzref;
y_pitch = data.ypitch; x_swf = data.yswfx; z_swf = data.yswfz; x_com = data.ycomx; z_com = data.ycomz;
q1ref = data.q1ref; q2ref = data.q2ref; q3ref = data.q3ref; q4ref = data.q4ref;
tau1 = data.tau1; tau2 = data.tau2; tau3 = data.tau3; tau4 = data.tau4;
vcom = data.vcom; vstatic = data.vstatic; vbody = data.vbody;
stf_ang_mom_mj = data.stf_ang_mom_mj; stf_ang_mom_pin = data.stf_ang_mom_pin;
x_ssp_curr_L = data.x_ssp_curr_L; L_ssp_curr_L = data.L_ssp_curr_L; x_ssp_impact_L = data.x_ssp_impact_L; L_ssp_impact_L = data.L_ssp_impact_L; x_ssp_impact_ref_L = data.x_ssp_impact_ref_L; L_ssp_impact_ref_L = data.L_ssp_impact_ref_L;
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
plot(t, x_swf)
plot(t, z_swf)
hold off
legend('x swf ref', 'z swf ref', 'x swf', 'z swf')
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

%% Plot LLIP Predictions
figure();
subplot(2, 1, 1)
hold on
plot(t, x_ssp_curr_L)
plot(t, x_ssp_impact_L)
plot(t, x_ssp_impact_ref_L)
hold off;
legend('x com', 'predicted x com impact', 'x com impact ref')
subplot(2, 1, 2)
hold on
plot(t, L_ssp_curr_L)
plot(t, L_ssp_impact_L)
plot(t, L_ssp_impact_ref_L)
hold off
legend('L com', 'predicted L com impact', 'L com impact ref')
sgtitle('LLIP Approximation')

%% Plot Outputs
figure();
subplot(3, 1, 1)
hold on
plot(t, y_pitchref)
plot(t, y_pitch)
hold off
legend("Pitch Ref", "Pitch")
subplot(3, 1, 2)

hold on;
plot(t, x_swf_ref)
plot(t, z_swf_ref)
plot(t, x_swf)
plot(t, z_swf)
hold off
legend("x swf ref", "z swf ref","x swf", "z swf")
subplot(3, 1, 3)
hold on
plot(t, x_com_ref)
plot(t, z_com_ref)
plot(t, x_com)
plot(t, z_com)
hold off
legend("x com ref", "z com ref","x com", "z com")
sgtitle('Output References')

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

%% Plot Feedforward Torque
figure();
hold on
plot(t, tau1)
plot(t, tau2)
plot(t, tau3)
plot(t, tau4)
hold off
legend('tau1', 'tau2', 'tau3', 'tau4')
title('Feedfoward Torque')

%% Plot CoM velocity approxes
figure();
hold on;
plot(t, vcom)
plot(t, vstatic)
plot(t, vbody)
hold off;
legend("CoM", "Static", "Body")
title("CoM Velocity Approximations")

%% Stance Foot Angular Mommentum
figure();
hold on
plot(t, stf_ang_mom_mj)
plot(t, stf_ang_mom_pin)
hold off
legend("MJC", "PIN")
title("Angular Momentum about Stance Foot")