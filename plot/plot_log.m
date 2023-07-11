close all;
clear;
clc;

%% Conifigure plots

% Specify what you want to plot

% Joint plot
plot_joints = true;

% Include joint position plots
plot_joint_pos = true;

% Include joint velocity plots
plot_joint_vel = true;

% Include joint torque plots
plot_joint_tor = false;

% Include plots for the arms
plot_arms = true;

%% Get log data

% Specify the relative location of the log file
file_name = "log.csv";

% Store the log data into a table
data = readtable("log.csv");

% Get rid of the final data line as it may be incomplete
data(end, :) = [];


%% Plot joint states
figure(1);

if(plot_joints == true)
    joint_cols = 4 + 3 * plot_arms;
    joint_rows = 2 * (plot_joint_pos + plot_joint_vel + plot_joint_tor);
    k = 0;
    
    if(plot_joint_pos == true)
        % Left joint positions
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_hip_yaw_pos);
        plot(data.t, data.left_hip_yaw_pos_ref);
        legend("State", "Ref");
        title("left hip yaw pos");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_hip_roll_pos);
        plot(data.t, data.left_hip_roll_pos_ref);
        legend("State", "Ref");
        title("left hip roll pos");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_hip_pitch_pos);
        plot(data.t, data.left_hip_pitch_pos_ref);
        legend("State", "Ref");
        title("left hip pitch pos");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_knee_pitch_pos);
        plot(data.t, data.left_knee_pitch_pos_ref);
        legend("State", "Ref");
        title("left knee pitch pos");
        
        if(plot_arms == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_shoulder_yaw_pos);
        plot(data.t, data.left_shoulder_yaw_pos_ref);
        legend("State", "Ref");
        title("left shoulder yaw pos");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_shoulder_pitch_pos);
        plot(data.t, data.left_shoulder_pitch_pos_ref);
        legend("State", "Ref");
        title("left shoulder pitch pos");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_elbow_pitch_pos);
        plot(data.t, data.left_elbow_pitch_pos_ref);
        legend("State", "Ref");
        title("left elbow pitch pos");
        end
    end

    % Left joint velocities
    if(plot_joint_vel == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_hip_yaw_vel);
        plot(data.t, data.left_hip_yaw_vel_ref);
        legend("State", "Ref");
        title("left hip yaw vel");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_hip_roll_vel);
        plot(data.t, data.left_hip_roll_vel_ref);
        legend("State", "Ref");
        title("left hip roll vel");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_hip_pitch_vel);
        plot(data.t, data.left_hip_pitch_vel_ref);
        legend("State", "Ref");
        title("left hip pitch vel");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_knee_pitch_vel);
        plot(data.t, data.left_knee_pitch_vel_ref);
        legend("State", "Ref");
        title("left knee pitch vel");
        
        if(plot_arms == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_shoulder_yaw_vel);
        plot(data.t, data.left_shoulder_yaw_vel_ref);
        legend("State", "Ref");
        title("left shoulder yaw vel");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_shoulder_pitch_vel);
        plot(data.t, data.left_shoulder_pitch_vel_ref);
        legend("State", "Ref");
        title("left shoulder pitch vel");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_elbow_pitch_vel);
        plot(data.t, data.left_elbow_pitch_vel_ref);
        legend("State", "Ref");
        title("left elbow pitch vel");
        end
    end

    % Left joint torques
    if(plot_joint_tor == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_hip_yaw_tor_ref);
        legend("State", "Ref");
        title("left hip yaw tor");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_hip_roll_tor_ref);
        legend("State", "Ref");
        title("left hip roll tor");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_hip_pitch_tor_ref);
        legend("State", "Ref");
        title("left hip pitch tor");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_knee_pitch_tor_ref);
        legend("State", "Ref");
        title("left knee pitch tor");
        
        if(plot_arms == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_shoulder_yaw_tor_ref);
        legend("State", "Ref");
        title("left shoulder yaw tor");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_shoulder_pitch_tor_ref);
        legend("State", "Ref");
        title("left shoulder pitch tor");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.left_elbow_pitch_tor_ref);
        legend("State", "Ref");
        title("left elbow pitch tor");
        end
    end
        
    % Right joint positions
    if(plot_joint_pos == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_hip_yaw_pos);
        plot(data.t, data.right_hip_yaw_pos_ref);
        legend("State", "Ref");
        title("right hip yaw pos");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_hip_roll_pos);
        plot(data.t, data.right_hip_roll_pos_ref);
        legend("State", "Ref");
        title("right hip roll pos");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_hip_pitch_pos);
        plot(data.t, data.right_hip_pitch_pos_ref);
        legend("State", "Ref");
        title("right hip pitch pos");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_knee_pitch_pos);
        plot(data.t, data.right_knee_pitch_pos_ref);
        legend("State", "Ref");
        title("right knee pitch pos");
        
        if(plot_arms == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_shoulder_yaw_pos);
        plot(data.t, data.right_shoulder_yaw_pos_ref);
        legend("State", "Ref");
        title("right shoulder yaw pos");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_shoulder_pitch_pos);
        plot(data.t, data.right_shoulder_pitch_pos_ref);
        legend("State", "Ref");
        title("right shoulder pitch pos");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_elbow_pitch_pos);
        plot(data.t, data.right_elbow_pitch_pos_ref);
        legend("State", "Ref");
        title("right elbow pitch pos");
        end
    end
        
    % Right joint velocities
    if(plot_joint_vel == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_hip_yaw_vel);
        plot(data.t, data.right_hip_yaw_vel_ref);
        legend("State", "Ref");
        title("right hip yaw vel");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_hip_roll_vel);
        plot(data.t, data.right_hip_roll_vel_ref);
        legend("State", "Ref");
        title("right hip roll vel");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_hip_pitch_vel);
        plot(data.t, data.right_hip_pitch_vel_ref);
        legend("State", "Ref");
        title("right hip pitch vel");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_knee_pitch_vel);
        plot(data.t, data.right_knee_pitch_vel_ref);
        legend("State", "Ref");
        title("right knee pitch vel");
        
        if(plot_arms == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_shoulder_yaw_vel);
        plot(data.t, data.right_shoulder_yaw_vel_ref);
        legend("State", "Ref");
        title("right shoulder yaw vel");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_shoulder_pitch_vel);
        plot(data.t, data.right_shoulder_pitch_vel_ref);
        legend("State", "Ref");
        title("right shoulder pitch vel");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_elbow_pitch_vel);
        plot(data.t, data.right_elbow_pitch_vel_ref);
        legend("State", "Ref");
        title("right elbow pitch vel");
        end
    end
        
    % Right joint torques
    if(plot_joint_tor == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_hip_yaw_tor_ref);
        legend("State", "Ref");
        title("right hip yaw tor");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_hip_roll_tor_ref);
        legend("State", "Ref");
        title("right hip roll tor");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_hip_pitch_tor_ref);
        legend("State", "Ref");
        title("right hip pitch tor");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_knee_pitch_tor_ref);
        legend("State", "Ref");
        title("right knee pitch tor");
        
        if(plot_arms == true)
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_shoulder_yaw_tor_ref);
        legend("State", "Ref");
        title("right shoulder yaw tor");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_shoulder_pitch_tor_ref);
        legend("State", "Ref");
        title("right shoulder pitch tor");
        
        k = k + 1;
        subplot(joint_rows, joint_cols, k);
        hold on;
        grid on;
        plot(data.t, data.right_elbow_pitch_tor_ref);
        legend("State", "Ref");
        title("right elbow pitch tor");
        end
    end
end