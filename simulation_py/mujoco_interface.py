import time
import numpy as np
import mujoco as mj
import mujoco.viewer

class MujocoInterface:

    class ContactData:

        def __init__(self, name_parent, name_child, id_parent, id_child):
            self.name_parent_geom = name_parent
            self.name_child_geom = name_child
            self.id_parent = id_parent
            self.id_child = id_child
            self.active = False
            self.force = np.zeros((3,))
            self.torque = np.zeros((3,))

        def setForce(self, force):
            self.force = force
        
        def setTorque(self, torque):
            self.torque = torque
        
        def setActive(self, active):
            self.active = active
            if self.active:
                self.force = np.zeros((3,))
                self.torque = np.zeros((3,))
    

    def __init__(self, xml_path: str, vis_rate: int=60, vis_enabled:bool=True):
        self.vis_enabled = vis_enabled
        self.setup(xml_path, vis_rate)

    def setup(self, xml_path: str, vis_rate: int=60) -> None:
        self.vis_rate = vis_rate
        # Create model and data objects
        self.mj_model = mj.MjModel.from_xml_path(xml_path)
        self.mj_data = mj.MjData(self.mj_model)
        mj.mj_forward(self.mj_model, self.mj_data)

        # Create and initialize viewer object
        if self.vis_enabled:
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            self.mj_viewer.sync()
            self.prevVisSync = time.time()

        # Populate the geom map
        self.geom_map = [self.mj_model.names + self.mj_model.name_geomadr[ii] for ii in range(self.mj_model.ngeom)]
        
        # Contact Map
        self.contact_map = {}
        for ii in range(self.mj_model.npair):
            contact_pair_name = self.mj_model.names + self.mj_model.name_pairadr[ii]

            id_parent = self.mj_model.pair_geom1[ii]
            id_child = self.mj_model.pair_geom2[ii]
            name_parent = self.mj_model.names + self.mj_model.name_geomadr[id_parent]
            name_child = self.mj_model.names + self.mj_model.name_geomadr[id_child]

            self.contact_map[contact_pair_name] = self.ContactData(name_parent, name_child, id_parent, id_child)

    def updateScene(self) -> None:
        if self.vis_enabled and self.mj_viewer.is_running():
            self.mj_viewer.sync()
        else:
            raise RuntimeError("Viewer has been closed!")

    def getBasePosition(self) -> np.ndarray:
        return self.mj_data.qpos[:3]

    def getBaseVelocity(self) -> np.ndarray:
        return self.mj_data.qvel[:3]
    
    def getCoMPosition(self) -> np.ndarray:
        # return (self.mj_data.subtree_com, self.mj_data.subtree_com[0, 0:3:2])
        return self.mj_data.subtree_com[0, 0:3:2]

    def getCoMVelocity(self) -> np.ndarray:
        mj.mj_subtreeVel(self.mj_model, self.mj_data)
        return (self.mj_data.subtree_linvel, self.mj_data.subtree_linvel[0, 0:3:2])

    def getJointPosition(self) -> np.ndarray:
        return self.mj_data.qpos[3:]

    def getJointVelocity(self) -> np.ndarray:
        return self.mj_data.qvel[3:]

    def setState(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.mj_data.qpos = qpos
        self.mj_data.qvel = qvel

    def getGenPosition(self) -> np.ndarray:
        return self.mj_data.qpos
    
    def getGenVelocity(self) -> np.ndarray:
        return self.mj_data.qvel

    # def setState(self, pos_base: np.ndarray, pos_joints: np.ndarray, vel_base: np.ndarray, vel_joints: np.ndarray) -> None:
    #     self.setState(np.hstack(pos_base, pos_joints), np.hstack(vel_base, vel_joints))

    def jointPosCmd(self, joint_pos_ref: np.ndarray) -> None:
        self.mj_data.ctrl[:4] = joint_pos_ref

    def jointVelCmd(self, joint_vel_ref: np.ndarray) -> None:
        self.mj_data.ctrl[4:8] = joint_vel_ref

    def jointTorCmd(self, tau_ref: np.ndarray) -> None:
        self.mj_data.ctrl[8:12] = tau_ref

    def getFootPos(self):
        return (self.mj_data.geom_xpos[12, ::2], self.mj_data.geom_xpos[7, ::2])

    def getContactForces(self):
        pass

    def step(self) -> None:
        mj.mj_step(self.mj_model, self.mj_data)

    def forward(self) -> None:
        mj.mj_forward(self.mj_model, self.mj_data)
    
    def resetContact(self) -> None:
        for (contact_pair_name, contact_data) in self.contact_map.items():
            contact_data.setActive(False)
    
    def time(self) -> float:
        return self.mj_data.time

    def getContact(self) -> None:
        self.resetContact()
        for ii in range(self.mj_data.ncon):
            id_parent_geom = self.mj_data.contact[ii].geom1
            id_child_geom = self.mj_data.contact[ii].geom2

            for (contact_pair_name, contact_data) in self.contact_map.items():
                if contact_data.id_parent == id_parent_geom and contact_data.id_child == id_child_geom:
                    generalizedForce = np.zeros((6,)) 
                    mj.mj_contactForce(self.mj_model, self.mj_data, ii, generalizedForce)

                    contact_force_contframe = generalizedForce[:3]
                    contact_torque_contframe = generalizedForce[3:]

                    R = self.mj_data.contact[ii].frame.reshape((3, 3))

                    contact_data.active = True
                    contact_data.force = R @ contact_force_contframe
                    contact_data.torque = R @ contact_torque_contframe

                    print(contact_data.force, contact_data.torque)

                    break



if __name__ == "__main__":
    import time
    mjInt = MujocoInterface("rsc/models/adam.xml")

    t0 = time.time()
    while True:
        
        mjInt.step()

        if time.time() - t0 >= 1 / 60:
            mjInt.updateScene()
            t0 = time.time()