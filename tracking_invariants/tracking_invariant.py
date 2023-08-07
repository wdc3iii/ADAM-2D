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

class TrackingInvariant:
    
    def __init__(self, v_ref, z_ref, pitch_ref, T_SSP, mass, useAngMomState=False, use_static_com=False, gravity_comp=True, use_task_space_ctrl=False):
        self.v_ref = v_ref
        self.z_ref = z_ref
        self.T_SSP = T_SSP
        
        self.controller = HLIPController(
            T_SSP, z_ref, urdf_path, mesh_path, mass, angMomState=useAngMomState,
            v_ref=v_ref, pitch_ref=pitch_ref, use_static_com=use_static_com, grav_comp=gravity_comp,
            use_task_space_ctrl=use_task_space_ctrl
        )

        self.adamKin = Kinematics(urdf_path, mesh_path)
        self.mjInt = MujocoInterface(xml_path, vis_enabled=False)

        q_ik = self.adamKin.getZeroPos()
        q_ik[Kinematics.GEN_POS_ID["P_LHP"]] = -0.4
        q_ik[Kinematics.GEN_POS_ID["P_RHP"]] = -0.4
        q_ik[Kinematics.GEN_POS_ID["P_LKP"]] = 0.8
        q_ik[Kinematics.GEN_POS_ID["P_RKP"]] = 0.8

        self.x_pre_ref = self.controller.calcPreImpactStateRef_HLIP(self.v_ref)
        self.u_ref = self.v_ref * self.T_SSP

    def S2S_sim(self, q0, qd0) -> tuple:
        self.mjInt.setState(q0, qd0)    
        startTime = self.mjInt.time()
        self.controller.reset()

        while True:
            t = self.mjInt.time() - startTime
            qpos = self.mjInt.getGenPosition()
            qvel = self.mjInt.getGenVelocity()   
            stfAngMom = self.mjInt.getSTFAngularMomentum()
            q_pos_ref, q_vel_ref, q_ff_ref, ddq_ref = self.controller.gaitController(qpos, qpos, qvel, t, self.mjInt.rightContact, self.mjInt.leftContact, stfAngMom)

            if not self.controller.cur_stf:
                break
            self.mjInt.jointPosCmd(q_pos_ref)
            self.mjInt.jointVelCmd(q_vel_ref)
            self.mjInt.jointTorCmd(q_ff_ref)

            self.mjInt.step()

        return qpos, qvel

        