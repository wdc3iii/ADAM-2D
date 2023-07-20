import yaml
import numpy as np
from plot.logger import Logger
from kinematics_py.adam_kinematics import Kinematics
from control_py.hlip_controller import HLIPController
from simulation_py.mujoco_interface import MujocoInterface

urdf_path = "rsc/models/adam2d_lightlimbs.urdf"
xml_path = "rsc/models/adam2d_lightlimbs.xml"
# urdf_path = "rsc/models/adam2d.urdf"
# xml_path = "rsc/models/adam2d.xml"
mesh_path = "rsc/models/"
log_path = "plot/log_main.csv"

def main():
    with open('rsc/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    use_static_com = config["use_static_com"]
    gravity_comp = config["gravity_comp"]
    pitch_ref = config["pitch_ref"]

    t_ref_list = config["ref_times"]
    z_ref_list = config["z_ref"]
    v_ref_list = config["v_ref"]
    T_SSP = config["T_SSP"]
    z_ref = 0.6
    v_ref = 0

    adamKin = Kinematics(urdf_path, mesh_path)
    mjInt = MujocoInterface(xml_path)

    q_pos_ref = adamKin.getZeroPos()
    q_pos_ref[Kinematics.GEN_POS_ID["P_LHP"]] = -0.4
    q_pos_ref[Kinematics.GEN_POS_ID["P_RHP"]] = -0.4
    q_pos_ref[Kinematics.GEN_POS_ID["P_LKP"]] = 0.8
    q_pos_ref[Kinematics.GEN_POS_ID["P_RKP"]] = 0.8

    
    controller = HLIPController(T_SSP, z_ref, urdf_path, mesh_path, v_ref=v_ref, pitch_ref=pitch_ref, use_static_com=use_static_com, grav_comp=gravity_comp)
    x_pre_ref = controller.calcPreImpactStateRef(v_ref)

    y_out_ref = np.zeros((Kinematics.N_OUTPUTS))
    y_out_ref[Kinematics.OUT_ID["PITCH"]] = pitch_ref
    y_out_ref[Kinematics.OUT_ID["SWF_POS_X"]] = 2 * x_pre_ref[0]
    y_out_ref[Kinematics.OUT_ID["SWF_POS_Z"]] = 0
    y_out_ref[Kinematics.OUT_ID["COM_POS_X"]] = x_pre_ref[0] # No control authority over x position
    y_out_ref[Kinematics.OUT_ID["COM_POS_Z"]] = z_ref

    q_pos_ref, _ = adamKin.solveIK(q_pos_ref, y_out_ref, True)

    # q_pos_ref[Kinematics.GEN_POS_ID["P_Z"]] += 0.71 - z_ref
    q_vel_ref = np.zeros((Kinematics.N_VEL_STATES,))

    mjInt.setState(q_pos_ref, q_vel_ref)
    mjInt.forward()
    ft_pos = mjInt.getFootPos()
    q_pos_ref[Kinematics.GEN_POS_ID['P_Z']] -= ft_pos[0][1] + 0.001
    mjInt.setState(q_pos_ref, q_vel_ref)

    logger = Logger(log_path, "t,x,z,pitch,q1,q2,q3,q4,xdot,zdot,pitchdot,q1dot,q2dot,q3dot,q4dot,q1ref,q2ref,q3ref,q4ref,q1dotref,q2dotref,q3dotref,q4dotref,tau1,tau2,tau3,tau4\n")


    ref_ind = 0
    while mjInt.viewerActive():

        t_vis = mjInt.time()

        while mjInt.time() - t_vis <= 1 / 60:
            t = mjInt.time()
            mjInt.getContact()

            if ref_ind < len(t_ref_list) and t >= t_ref_list[ref_ind]:
                v_ref = v_ref_list[ref_ind]
                z_ref = z_ref_list[ref_ind]
                ref_ind += 1
                print(f"Reference Changed: v = {v_ref}, z = {z_ref}")
                controller.setV_ref(v_ref)
                controller.setZ_ref(z_ref)

            qpos = mjInt.getGenPosition()
            qvel = mjInt.getGenVelocity()
            q_pos_ref, q_vel_ref, q_ff_ref = controller.gaitController(qpos, qpos, qvel, t, mjInt.rightContact, mjInt.leftContact)

            mjInt.jointPosCmd(q_pos_ref)
            mjInt.jointVelCmd(q_vel_ref)
            mjInt.jointTorCmd(q_ff_ref)

            log_data = np.hstack((
                t, qpos, qvel, q_pos_ref, q_vel_ref, q_ff_ref
            ))
            logger.write(log_data)

            mjInt.step()

        mjInt.updateScene()
    
    logger.close()


if __name__ == "__main__":
    main()