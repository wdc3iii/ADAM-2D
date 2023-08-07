import yaml
import numpy as np
import time
from plot.logger import Logger
from kinematics_py.adam_kinematics import Kinematics
from control_py.hlip_controller import HLIPController
from simulation_py.mujoco_interface import MujocoInterface

mesh_path = "rsc/models/"
log_path = "plot/log_main.csv"
urdf_path = "rsc/models/adam2d.urdf"
xml_path = "rsc/models/adam2d.xml"

def main():
    with open('rsc/track_inv_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    use_static_com = config["use_static_com"]
    gravity_comp = config["gravity_comp"]
    useAngMomState = config["use_ang_mom_state"]
    use_task_space_ctrl = config["use_task_space_ctrl"]
    pitch_ref = config["pitch_ref"]

    z_ref = config["z_ref"]
    v_ref = config["v_ref"]
    T_SSP = config["T_SSP"]

    adamKin = Kinematics(urdf_path, mesh_path)
    mjInt = MujocoInterface(xml_path)

    q_pos_ref = adamKin.getZeroPos()
    q_pos_ref[Kinematics.GEN_POS_ID["P_LHP"]] = -0.4
    q_pos_ref[Kinematics.GEN_POS_ID["P_RHP"]] = -0.4
    q_pos_ref[Kinematics.GEN_POS_ID["P_LKP"]] = 0.8
    q_pos_ref[Kinematics.GEN_POS_ID["P_RKP"]] = 0.8

    
    controller = HLIPController(
        T_SSP, z_ref, urdf_path, mesh_path, mjInt.mass, angMomState=useAngMomState,
        v_ref=v_ref, pitch_ref=pitch_ref, use_static_com=use_static_com, grav_comp=gravity_comp,
        use_task_space_ctrl=use_task_space_ctrl
    )
    x_pre_ref = controller.calcPreImpactStateRef_HLIP(v_ref)
    u_ref = v_ref * T_SSP

    y_out_ref = np.zeros((Kinematics.N_OUTPUTS))
    y_out_ref[Kinematics.OUT_ID["PITCH"]] = pitch_ref
    y_out_ref[Kinematics.OUT_ID["SWF_POS_X"]] = u_ref
    y_out_ref[Kinematics.OUT_ID["SWF_POS_Z"]] = 0
    y_out_ref[Kinematics.OUT_ID["COM_POS_X"]] = x_pre_ref[0]
    y_out_ref[Kinematics.OUT_ID["COM_POS_Z"]] = z_ref

    q_pos_ref, _ = adamKin.solveIK(q_pos_ref, y_out_ref, True)

    yd_out_ref = np.array([
        0, 0, controller.swf_pos_z_poly.evalPoly(T_SSP, 1), x_pre_ref[1], 0, 0, 0
    ])
    Jy_out_ref = np.vstack((np.array([0, 0, 1, 0, 0, 0, 0]), adamKin.getSWFJacobian(q_pos_ref, True)[0:3:2, :], adamKin.getCoMJacobian(q_pos_ref)[0:3:2, :], adamKin.getSTFJacobian(q_pos_ref, True)[0:3:2, :]))

    q_vel_ref = np.linalg.inv(Jy_out_ref) @ yd_out_ref

    mjInt.setState(q_pos_ref, q_vel_ref)    
    mjInt.forward()
    ft_pos = mjInt.getFootPos()
    q_pos_ref[Kinematics.GEN_POS_ID['P_Z']] -= ft_pos[0][1] - adamKin.getContactOffset(q_pos_ref, True)
    mjInt.setState(q_pos_ref, q_vel_ref)
    mjInt.forward()
    logger = Logger(log_path, "t,x,z,pitch,q1,q2,q3,q4,xdot,zdot,pitchdot,q1dot,q2dot,q3dot,q4dot,xddot,zddot,pitchddot,q1ddot,q2ddot,q3ddot,q4ddot,q1ref,q2ref,q3ref,q4ref,q1dotref,q2dotref,q3dotref,q4dotref," + \
                    "tau1,tau2,tau3,tau4,m11,m12,m13,m14,m15,m16,m17,m21,m22,m23,m24,m25,m26,m27,m31,m32,m33,m34,m35,m36,m37,m41,m42,m43,m44,m45,m46,m47,m51,m52,m53,m54,m55,m56,m57,m61,m62,m63,m64,m65,m66,m67," + \
                    "m71,m72,m73,m74,m75,m76,m77,h1,h2,h3,h4,h5,h6,h7,F00,F01,F02,T00,T01,T02,F10,F11,F12,T10,T11,T12,J1h11,J1h12,J1h13,J1h14,J1h15,J1h16,J1h17,J1h21,J1h22,J1h23,J1h24,J1h25,J1h26,J1h27," + \
                    "J2h11,J2h12,J2h13,J2h14,J2h15,J2h16,J2h17,J2h21,J2h22,J2h23,J2h24,J2h25,J2h26,J2h27\n")


    mjInt.updateScene()
    # time.sleep(30)
    # print("Init")
    
    ref_ind = 0
    while mjInt.viewerActive():

        t_vis = mjInt.time()

        while mjInt.time() - t_vis <= 1 / 60:
            t = mjInt.time()
            
            # mjInt.updateScene()
            # time.sleep(3)
            # print(t)
            
            mjInt.getContact()

            qpos = mjInt.getGenPosition()
            qvel = mjInt.getGenVelocity()   
            qacc = mjInt.getGenAccel()
            M_mjc, H_mjc, Jh1_mjc, Jh2_mjc, F_mjc = mjInt.getDynamics()
            stfAngMom = mjInt.getSTFAngularMomentum()
            q_pos_ref, q_vel_ref, q_ff_ref, ddq_ref = controller.gaitController(qpos, qpos, qvel, t, mjInt.rightContact, mjInt.leftContact, stfAngMom)

            mjInt.jointPosCmd(q_pos_ref)
            mjInt.jointVelCmd(q_vel_ref)
            mjInt.jointTorCmd(q_ff_ref)
            contacts = mjInt.getContactForces()
            if not controller.cur_stf:
                break
            log_data = np.hstack((
                t, qpos, qvel, qacc, q_pos_ref, q_vel_ref, q_ff_ref, M_mjc.reshape((-1,)), H_mjc, contacts[0].force, contacts[0].torque, contacts[1].force, contacts[1].torque, Jh1_mjc.reshape((-1,)), Jh2_mjc.reshape((-1,))
            ))
            logger.write(log_data)

            
            mjInt.step()


        mjInt.updateScene()

        if not controller.cur_stf:
            break
    
    logger.close()



if __name__ == "__main__":
    main()