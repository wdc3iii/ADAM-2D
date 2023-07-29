close all;
clear;
clc;

%% Get log data

% Specify the relative location of the log file
file_name_ctrl = "log_ctrl.csv";
file_name_main = "log_main.csv";
% Store the log data into a table
data = readtable(file_name_ctrl);
data_MJC = readtable(file_name_main);

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
grfx = data.grfx; grfz = data.grfz;
tau1_gc = data.tau1_gc; tau2_gc = data.tau2_gc; tau3_gc = data.tau3_gc; tau4_gc = data.tau4_gc;
tau1_tsc = data.tau1_tsc; tau2_tsc = data.tau2_tsc; tau3_tsc = data.tau3_tsc; tau4_tsc = data.tau4_tsc;
ddx = data.ddx; ddz = data.ddz; ddtheta = data.ddtheta; ddq1 = data.ddq1; ddq2 = data.ddq2; ddq3 = data.ddq3; ddq4 = data.ddq4;
deltaepitch = data.deltaepitch; deltaeswfx = data.deltaeswfx; deltaeswfz = data.deltaeswfz; deltaecomz = data.deltaecomz;
obj_val = data.obj_val;
H = [data.h1, data.h2, data.h3, data.h4, data.h5, data.h6, data.h7];
Md = [data.m11, data.m12, data.m13, data.m14, data.m15, data.m16, data.m17, data.m22, data.m23, data.m24, data.m25, data.m26, data.m27,data.m33, data.m34, data.m35, data.m36, data.m37,data.m44, data.m45, data.m46, data.m47,data.m55, data.m56, data.m57,data.m66, data.m67, data.m77];
M = zeros(7,7,size(Md, 1));
M(1, :, :) = Md(:, 1:7)';
M(2, 2:end, :) = Md(:, 8:13)';
M(3, 3:end, :) = Md(:, 14:18)';
M(4, 4:end, :) = Md(:, 19:22)';
M(5, 5:end, :) = Md(:, 23:25)';
M(6, 6:end, :) = Md(:, 26:27)';
M(7, 7, :) = Md(:, 28)';
M(:, 1, :) = Md(:, 1:7)';
M(2:end, 2, :) = Md(:, 8:13)';
M(3:end, 3, :) = Md(:, 14:18)';
M(4:end, 4, :) = Md(:, 19:22)';
M(5:end, 5, :) = Md(:, 23:25)';
M(6:end, 6, :) = Md(:, 26:27)';
J = zeros(2, 7, size(Md, 1));
J(1, 1, :) = data.Jh11;
J(1, 2, :) = data.Jh12;
J(1, 3, :) = data.Jh13;
J(1, 4, :) = data.Jh14;
J(1, 5, :) = data.Jh15;
J(1, 6, :) = data.Jh16;
J(1, 7, :) = data.Jh17;
J(2, 1, :) = data.Jh21;
J(2, 2, :) = data.Jh22;
J(2, 3, :) = data.Jh23;
J(2, 4, :) = data.Jh24;
J(2, 5, :) = data.Jh25;
J(2, 6, :) = data.Jh26;
J(2, 7, :) = data.Jh27;


H_mjc = [data_MJC.h1, data_MJC.h2, data_MJC.h3, data_MJC.h4, data_MJC.h5, data_MJC.h6, data_MJC.h7];
Md_mjc = [data_MJC.m11, data_MJC.m12, data_MJC.m13, data_MJC.m14, data_MJC.m15, data_MJC.m16, data_MJC.m17, data_MJC.m22, data_MJC.m23, data_MJC.m24, data_MJC.m25, data_MJC.m26, data_MJC.m27,data_MJC.m33, data_MJC.m34, data_MJC.m35, data_MJC.m36, data_MJC.m37,data_MJC.m44, data_MJC.m45, data_MJC.m46, data_MJC.m47,data_MJC.m55, data_MJC.m56, data_MJC.m57,data_MJC.m66, data_MJC.m67, data_MJC.m77];
M_mjc = zeros(7,7,size(Md_mjc, 1));
M_mjc(1, :, :) = Md_mjc(:, 1:7)';
M_mjc(2, 2:end, :) = Md_mjc(:, 8:13)';
M_mjc(3, 3:end, :) = Md_mjc(:, 14:18)';
M_mjc(4, 4:end, :) = Md_mjc(:, 19:22)';
M_mjc(5, 5:end, :) = Md_mjc(:, 23:25)';
M_mjc(6, 6:end, :) = Md_mjc(:, 26:27)';
M_mjc(:, 1, :) = Md_mjc(:, 1:7)';
M_mjc(2:end, 2, :) = Md_mjc(:, 8:13)';
M_mjc(3:end, 3, :) = Md_mjc(:, 14:18)';
M_mjc(4:end, 4, :) = Md_mjc(:, 19:22)';
M_mjc(5:end, 5, :) = Md_mjc(:, 23:25)';
M_mjc(6:end, 6, :) = Md_mjc(:, 26:27)';
ddx_mjc = data_MJC.xddot; ddz_mjc = data_MJC.zddot; ddp_mjc = data_MJC.pitchddot; 
ddq1_mjc = data_MJC.q1ddot;ddq2_mjc = data_MJC.q2ddot;ddq3_mjc = data_MJC.q3ddot;ddq4_mjc = data_MJC.q4ddot;

F1x = data_MJC.F00; F1z = data_MJC.F02;
F2x = data_MJC.F10; F2z = data_MJC.F12;

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

step_inds = find(abs(diff(x_curr)) > 0.005);
step_times = t(step_inds);

xt = [];
tt = [];

zref = 0.7;
% m = 14.607895999999995;
m = 9.224171599999998;
lam = sqrt(9.81 / zref);
A = [0 1; lam ^2 0];

for ii = 1:size(step_inds)
    x0 = [x_curr(step_inds(ii) + 1); v_curr(step_inds(ii) + 1)];
    [t1, y] = ode45(@(t, x) A * x, [0, 0.3], x0);
    xt = [xt; y];
    tt = [tt; t1 + t(step_inds(ii) + 1)];
end
subplot(2, 1, 1)
hold on
plot(tt, xt(:, 1))
hold off
subplot(2, 1, 2)
hold on
plot(tt, xt(:, 2))
hold off

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

%% Grav Comp vs TSC
figure()
hold on
plot(t, tau1_gc, '--')
plot(t, tau2_gc, '--')
plot(t, tau3_gc, '--')
plot(t, tau4_gc, '--')
plot(t, tau1_tsc)
plot(t, tau2_tsc)
plot(t, tau3_tsc)
plot(t, tau4_tsc)
hold off
legend("Tau1 gc", "Tau2 gc", "Tau3 gc", "Tau4 gc", "Tau1 tsc", "Tau2 tsc", "Tau3 tsc", "Tau4 tsc")



%% DeltaE (QP objective)
figure()
subplot(2, 1, 1)
hold on
plot(t, deltaepitch)
plot(t, deltaeswfx)
plot(t, deltaeswfz)
plot(t, deltaecomz)
hold off
legend("deltaEPitch", "deltaESWFX", "deltaESWFZ", "deltaECOMZ")
subplot(2, 1, 2)
plot(t, obj_val)
legend("Objective Value")

%% ddq

Tt = min(size(t, 1), size(H_mjc, 1));

figure()
subplot(2, 1, 1)
hold on
plot(t(1:Tt), ddx(1:Tt))
plot(t(1:Tt), ddz(1:Tt))
plot(t(1:Tt), ddtheta(1:Tt))
plot(t(1:Tt), ddx_mjc(1:Tt), "--")
plot(t(1:Tt), ddz_mjc(1:Tt), "--")
plot(t(1:Tt), ddp_mjc(1:Tt), "--")
hold off
legend("ddx", "ddz", "ddtheta","ddxmjc", "ddzmjc", "ddthetamjc")
subplot(2, 1, 2)
hold on
plot(t(1:Tt), ddq1(1:Tt))
plot(t(1:Tt), ddq2(1:Tt))
plot(t(1:Tt), ddq3(1:Tt))
plot(t(1:Tt), ddq4(1:Tt))
plot(t(1:Tt), ddq1_mjc(1:Tt), "--")
plot(t(1:Tt), ddq2_mjc(1:Tt), "--")
plot(t(1:Tt), ddq3_mjc(1:Tt), "--")
plot(t(1:Tt), ddq4_mjc(1:Tt), "--")
hold off
legend("ddq1", "ddq2", "ddq3", "ddq4", "ddq1mjc", "ddq2mjc", "ddq3mjc", "ddq4mjc")

%% Ground Reaction Force
figure();
subplot(2, 1, 1)
hold on
plot(t(1:Tt), grfx(1:Tt))
plot(t(1:Tt), -F1x(1:Tt))
plot(t(1:Tt), -F2x(1:Tt))
hold off
legend("TSC", "F1mjc", "F2mjc")
ylabel("X GRF")
subplot(2, 1, 2)
hold on
plot(t(1:Tt), grfz(1:Tt))
plot(t(1:Tt), -F1z(1:Tt))
plot(t(1:Tt), -F2z(1:Tt))
hold off
legend("TSC", "F1mjc", "F2mjc")
ylabel("Z GRF")

%% H
figure()
for ii = 1:7
    subplot(4, 2, ii)
    hold on
    plot(t(1:Tt), H(1:Tt, ii))
    plot(t(1:Tt), H_mjc(1:Tt, ii))
    hold off
    fprintf("Hdiff: %e\n", max(abs(H(1:Tt, ii)-H_mjc(1:Tt, ii))))
end

%% M
figure()
for r = 1:7
    for c = 1:7
        subplot(7, 7, 7 * (r - 1) + c)
        hold on
        plot(t(1:Tt), squeeze(M(r, c, 1:Tt)))
        plot(t(1:Tt), squeeze(M_mjc(r, c, 1:Tt)))
        hold off
        fprintf("Mdiff: %e\n", max(abs(M(r, c, 1:Tt) - M_mjc(r, c, 1:Tt))))
    end
end

%% Check Dynamics
B = [zeros(3, 4); eye(4)];
delta_ddq = zeros(size(t));
for ii = 1:size(M, 3)
    Mt = squeeze(M(:, :, ii));
    ddq = [ddx(ii); ddz(ii); ddtheta(ii); ddq1(ii); ddq2(ii); ddq3(ii); ddq4(ii)];
    Ht = H(ii, :)';
    Jt = J(:, :, ii);
    Ft = [grfx(ii); grfz(ii)];
    taut = [tau1_tsc(ii); tau2_tsc(ii); tau3_tsc(ii); tau4_tsc(ii)];
    ddq_fd = Mt \ (-Ht + B * taut + Jt' * Ft);
    delta_ddq(ii) = norm(ddq - ddq_fd);
    if delta_ddq(ii) > 1
        disp("uhoh")
    end
end
figure()
plot(t, delta_ddq)