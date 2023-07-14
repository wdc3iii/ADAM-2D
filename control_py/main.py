import yaml
import numpy as np
from kinematics_py.adam_kinematics import Kinematics
from control_py.hlip_controller import HLIPController
from simulation_py.mujoco_interface import MujocoInterface
from plot.logger import Logger

urdf_path = "rsc/models/adam.urdf"
mesh_path = "rsc/models/"
xml_path = "rsc/models/adam.xml"
log_path = "plot/log.csv"

def main():
    with open('rsc/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(config)

    use_static_com = config["use_static_com"]
    pitch_ref = config["pitch_ref"]

    t_ref_list = config["ref_times"]
    z_ref_list = config["z_ref"]
    v_ref_list = config["v_ref"]
    T_SSP = config["T_SSP"]
    z_ref = 0.6
    v_ref = 0

    adamKin = Kinematics(urdf_path, mesh_path)
    mjInt = MujocoInterface(xml_path)

    stanceFoot = True

    q_pos_ref = adamKin.getZeroPos()
    

    controller = HLIPController(T_SSP, z_ref, urdf_path, mesh_path, v_ref=v_ref, pitch_ref=pitch_ref, use_static_com=use_static_com)
    x_pre_ref = controller.calcPreImpactStateRef(v_ref)

    y_out_ref = np.zeros((Kinematics.N_OUTPUTS))
    y_out_ref[Kinematics.OUT_ID["PITCH"]] = pitch_ref
    y_out_ref[Kinematics.OUT_ID["SWF_POS_X"]] = 2 * x_pre_ref[0]
    y_out_ref[Kinematics.OUT_ID["SWF_POS_Z"]] = 0
    y_out_ref[Kinematics.OUT_ID["COM_POS_X"]] = x_pre_ref[0] # No control authority over x position
    y_out_ref[Kinematics.OUT_ID["COM_POS_Z"]] = z_ref

    q_pos_ref[Kinematics.GEN_POS_ID["P_Z"]] = z_ref + 0.06
    mj_q_pos_ref = adamKin.convertGenPosPINtoMJC(q_pos_ref)
    q_vel_ref = np.zeros((Kinematics.N_VEL_STATES,))
    mj_q_vel_ref = adamKin.convertGenVelPintoMJC(q_vel_ref)

    logger = Logger(log_path)

    ref_ind = 0
    while True:

        t_vis = mjInt.time()

        while mjInt.time() - t_vis <= 1 / 60:
            t = mjInt.time()

            if ref_ind < t_ref_list.size and t >= t_ref_list[ref_ind]:
                v_ref = v_ref_list[ref_ind]
                z_ref = z_ref_list[ref_ind]
                ref_ind += 1
                print(f"Reference Changed: v = {v_ref}, z = {z_ref}")
            logger.write(log_data)

        mjInt.updateScene()




if __name__ == "__main__":
    main()