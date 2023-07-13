#include "../include/mujoco_interface.h"

// Generic Mujoco functions and variables

MujocoInterface::MujocoInterface(){}

MujocoInterface::~MujocoInterface(){}

void MujocoInterface::MujocoSetup(const char file_name[]) {
    char error_msg[1000] = "Failed to load binary model";

    MJ_MODEL_PTR = mj_loadXML(file_name, 0, error_msg, 1000);

    if (!MJ_MODEL_PTR) {
        mju_error_s("Load model error: %s", error_msg);
    } else {
        std::cout << "Model was loaded successfully" << std::endl;
    }

    MJ_DATA_PTR = mj_makeData(MJ_MODEL_PTR);

    // init GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    window = glfwCreateWindow(1244, 700, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&MJ_CAMERA);
    mjv_defaultOption(&MJ_OPTIONS);
    mjv_defaultScene(&MJ_SCENE);
    mjr_defaultContext(&MJ_CONTEXT);
    mjv_makeScene(MJ_MODEL_PTR, &MJ_SCENE, 2000);                // space for 2000 objects
    mjr_makeContext(MJ_MODEL_PTR, &MJ_CONTEXT, mjFONTSCALE_150);   // model-specific context

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // Set defualt camera view
    double arr_view[] = {90, -5, 5, 0.012768, -0.000000, 1.254336};
    MJ_CAMERA.azimuth = arr_view[0];
    MJ_CAMERA.elevation = arr_view[1];
    MJ_CAMERA.distance = arr_view[2];
    MJ_CAMERA.lookat[0] = arr_view[3];
    MJ_CAMERA.lookat[1] = arr_view[4];
    MJ_CAMERA.lookat[2] = arr_view[5];

    // Populate the geom map
    for(int i = 0; i < MJ_MODEL_PTR->ngeom; i++) {
        this->geom_map[i] = MJ_MODEL_PTR->names + MJ_MODEL_PTR->name_geomadr[i];
    }

    // Populate the contact pair map
    for(int i = 0; i < MJ_MODEL_PTR->npair; i++) {
        std::string contact_pair_name = MJ_MODEL_PTR->names + MJ_MODEL_PTR->name_pairadr[i];

        int id_parent_geom = MJ_MODEL_PTR->pair_geom1[i];

        int id_child_geom = MJ_MODEL_PTR->pair_geom2[i];

        std::string name_parent_geom = geom_map[id_parent_geom];

        std::string name_child_geom = geom_map[id_child_geom];

        ContactData contact_data(name_parent_geom, name_child_geom, id_parent_geom, id_child_geom);

        this->contact_pair_map[contact_pair_name] = contact_data;
    }
}

void MujocoInterface::MujocoShutdown() {
    // free visualization storage
    mjv_freeScene(&MJ_SCENE);
    mjr_freeContext(&MJ_CONTEXT);

    // free MuJoCo model and data, deactivate
    mj_deleteData(MJ_DATA_PTR);
    mj_deleteModel(MJ_MODEL_PTR);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif
}

void MujocoInterface::UpdateScene() {
    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
    mjv_updateScene(MJ_MODEL_PTR, MJ_DATA_PTR, &(MJ_OPTIONS), NULL, &MJ_CAMERA, mjCAT_ALL, &MJ_SCENE);
    mjr_render(viewport, &MJ_SCENE, &MJ_CONTEXT);

    // Make the camera follow the robot
    MJ_CAMERA.lookat[0] = MJ_DATA_PTR->qpos[0];
    MJ_CAMERA.lookat[1] = MJ_DATA_PTR->qpos[1];

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
}

void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    // backspace: reset simulation
    if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
        mj_resetData(MJ_MODEL_PTR, MJ_DATA_PTR);
        mj_forward(MJ_MODEL_PTR, MJ_DATA_PTR);
    }
}

void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    // update button state
    BUTTON_LEFT =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    BUTTON_MIDDLE = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    BUTTON_RIGHT =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &BUTTON_LAST_X, &BUTTON_LAST_Y);
}

void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    // no buttons down: nothing to do
    if (!BUTTON_LEFT && !BUTTON_MIDDLE && !BUTTON_RIGHT) {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - BUTTON_LAST_X;
    double dy = ypos - BUTTON_LAST_Y;
    BUTTON_LAST_X = xpos;
    BUTTON_LAST_Y = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (BUTTON_RIGHT) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (BUTTON_LEFT) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }
    // move camera
    mjv_moveCamera(MJ_MODEL_PTR, action, dx/height, dy/height, &MJ_SCENE, &MJ_CAMERA);
}

void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(MJ_MODEL_PTR, mjMOUSE_ZOOM, 0, -0.05*yoffset, &MJ_SCENE, &MJ_CAMERA);
}


// Robot specific functions

Eigen::Vector<double, 3> MujocoInterface::GetBasePositions() {
    Eigen::Vector<double, 3> q_base;

    // Get x_world
    q_base(0) = MJ_DATA_PTR->qpos[0];
    
    // Get z_world
    q_base(1) = MJ_DATA_PTR->qpos[1];

    // Get rotation_world
    q_base(2) = MJ_DATA_PTR->qpos[2];

    return q_base;
}

Eigen::Vector<double, 4> MujocoInterface::GetJointPositions() {
    Eigen::Vector<double, 4> q_joint;

    for (int i = 0; i < 4; i++) {
        // Obtain the joint angles (+1 because of the quaternion)
        q_joint(i) = MJ_DATA_PTR->qpos[3 + i];
    }

    return q_joint;
}

Eigen::Vector<double, 3> MujocoInterface::GetBaseVelocities() {
    Eigen::Vector<double, 3> q_d_base;

    // Linear x velocity in the world frame
    q_d_base(0) = MJ_DATA_PTR->qvel[0]; 

    // Linear z velocity in the world frame
    q_d_base(1) = MJ_DATA_PTR->qvel[1];

    // Angular y velocity in the world frame
    q_d_base(2) = MJ_DATA_PTR->qvel[2];

    return q_d_base;
}

Eigen::Vector<double, 4> MujocoInterface::GetJointVelocities() {
    Eigen::Vector<double, 4> q_d_joint;

    for (int i = 0; i < 4; i++) {
        // Obtain the joint velocities
        q_d_joint(i) = MJ_DATA_PTR->qvel[3 + i];
    }

    return q_d_joint;
}

void MujocoInterface::SetState(Eigen::Vector<double, 3> q_base, Eigen::Vector<double, 3> q_d_base, Eigen::Vector<double, 4> q_joint, Eigen::Vector<double, 4> q_d_joint) {
    // Base pose
    MJ_DATA_PTR->qpos[0] = q_base(0);
    MJ_DATA_PTR->qpos[1] = q_base(1);
    MJ_DATA_PTR->qpos[2] = q_base(2);

    // Base velocity
    MJ_DATA_PTR->qvel[0] = q_d_base(0);
    MJ_DATA_PTR->qvel[1] = q_d_base(1);
    MJ_DATA_PTR->qvel[2] = q_d_base(2);

    // Set the joint positions and velocities
    for (int i = 0; i < 4; i++) {
        // Set joint position
        MJ_DATA_PTR->qpos[3 + i] = q_joint(i);

        // Set joint velocity
        MJ_DATA_PTR->qvel[3 + i] = q_d_joint(i);
    }
}

void MujocoInterface::SetState(Eigen::Vector<double, 7> q_pos, Eigen::Vector<double, 7> q_vel) {
    for (int i = 0; i < 7; i++) {
        MJ_DATA_PTR->qpos[i] = q_pos(i);
        MJ_DATA_PTR->qvel[i] = q_vel(i);
    }
}

void MujocoInterface::JointPosCmd(Eigen::Vector<double, 4> joint_pos_ref) {
    for (int i = 0; i < 4; i++) {
        MJ_DATA_PTR->ctrl[i] = joint_pos_ref(i);
    }
}

void MujocoInterface::JointVelCmd(Eigen::Vector<double, 4> joint_vel_ref) {
    for (int i = 0; i < 4; i++) {
        MJ_DATA_PTR->ctrl[i + 4] = joint_vel_ref(i);
    }
}

void MujocoInterface::JointTorCmd(Eigen::Vector<double, 4> joint_torque_ref) {
    for (int i = 0; i < 4; i++) {
        MJ_DATA_PTR->ctrl[i + 2 * 4] = joint_torque_ref(i);
    }
}

Eigen::Vector<double, 7> MujocoInterface::GetGeneralizedPos() {
    Eigen::Vector<double, 7> q_gen_pos = Eigen::Vector<double, 7>::Zero();

    for (int i = 0; i < 7; i++) {
        q_gen_pos(i) = MJ_DATA_PTR->qpos[i];
    }

    return q_gen_pos;
}

Eigen::Vector<double, 7> MujocoInterface::GetGeneralizedVel() {
    Eigen::Vector<double, 7> q_gen_vel = Eigen::Vector<double, 7>::Zero();

    for (int i = 0; i < 7; i++) {
        q_gen_vel(i) = MJ_DATA_PTR->qvel[i];
    }

    return q_gen_vel;
}

void MujocoInterface::PropagateDynamics() {
    mj_step(MJ_MODEL_PTR, MJ_DATA_PTR);
}

void MujocoInterface::PrintContactForce()
{
    // Contact test
    double result[6] = {0,0,0,0,0,0};
    mj_contactForce(MJ_MODEL_PTR, MJ_DATA_PTR, 2, result);
    for(int i = 0; i < 6; i++) {
        std::cout << result[i] << "\t";
    }
    std::cout << std::endl;
    
}

std::map<std::string, ContactData> MujocoInterface::GetContactData() {
    // Initialize all the contacts to false and all forces and torques to zero for all contact pairs
    for (auto contact_pair = contact_pair_map.begin(); contact_pair != contact_pair_map.end(); ++contact_pair) {
        contact_pair->second.contact_active = false;
        contact_pair->second.force = Eigen::Vector3d::Zero();
        contact_pair->second.torque = Eigen::Vector3d::Zero();
    }   
    
    // Loop through all the detected contacts at the current time
    for (int i = 0; i < MJ_DATA_PTR->ncon; i++) {
        // Get the parent geom id of contact #i
        int id_parent_geom = MJ_DATA_PTR->contact[i].geom1;

        // Get the child geom id of contact #i
        int id_child_geom = MJ_DATA_PTR->contact[i].geom2;

        // Loop through all the contact pairs to find out what contact pair the detected contact corresponds to 
        for (auto contact_pair = contact_pair_map.begin(); contact_pair != contact_pair_map.end(); ++contact_pair) {
            // In order to correspond to specifc contact pair the child and parent geoms must match that of the contact pair
            if ((contact_pair->second.id_parent_geom == id_parent_geom) && (contact_pair->second.id_child_geom == id_child_geom)) {
                // Create an array to stor the contact forces and torques
                double generalized_forces[6] = {0,0,0,0,0,0};

                // Obtain the contact forces and torques from Mujoco
                mj_contactForce(MJ_MODEL_PTR, MJ_DATA_PTR, i, generalized_forces);

                Eigen::Vector3d contact_forces;
                contact_forces(0) = generalized_forces[0];
                contact_forces(1) = generalized_forces[1];
                contact_forces(2) = generalized_forces[2];

                Eigen::Vector3d contact_torques;
                contact_torques(0) = generalized_forces[3];
                contact_torques(1) = generalized_forces[4];
                contact_torques(2) = generalized_forces[5];

                // Obtain the rotation matrix between the contact frame and world frame
                Eigen::Matrix3d R = Eigen::Matrix3d::Zero();

                Eigen::Vector<double, 9> r_contact_frame;
                for (int r = 0; r < 9; r++) {
                    r_contact_frame(r) = MJ_DATA_PTR->contact[i].frame[r];
                }

                R.block<1, 3>(0, 0) = r_contact_frame.block<3, 1>(0, 0).transpose();
                R.block<1, 3>(1, 0) = r_contact_frame.block<3, 1>(3, 0).transpose();
                R.block<1, 3>(2, 0) = r_contact_frame.block<3, 1>(6, 0).transpose();

                //std::cout << R.transpose() << "\n" << std::endl;

                // Set the contact to active
                contact_pair->second.contact_active = true;

                // Set the contact forces
                contact_pair->second.force = R.transpose() * contact_forces;
                
                // Set the contact torques
                contact_pair->second.torque = R.transpose() * contact_torques;

                // A contact can only belong to one contact pair so we do not need to check the remaining contact pairs
                break;
            }
        }
    }

    return contact_pair_map;
}

Eigen::Vector<double, 2> MujocoInterface::GetComPos() {
    mj_forward(MJ_MODEL_PTR, MJ_DATA_PTR);
    std::cout << MJ_DATA_PTR->xipos << std::endl;
}

Eigen::Vector<double, 2> MuJoCoInterface::GetSWFootPos() {
    mj_forward(MJ_MODEL_PTR, MJ_DATA_PTR);
    std::cout << MJ_DATA_PTR->xpos << std::endl;
}