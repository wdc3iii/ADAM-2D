#ifndef mujoco_types
#define mujoco_types

#include <Eigen/Dense>
#include "stdio.h"
#include <iostream>

class ContactData {
    public: std::string name_parent_geom;
    public: std::string name_child_geom;
    public: int id_parent_geom;
    public: int id_child_geom;
    public: bool contact_active;
    public: Eigen::Vector3d force;
    public: Eigen::Vector3d torque;

    ContactData() {
        name_parent_geom = "";
        name_child_geom = "";
        id_parent_geom = -1;
        id_child_geom = -1;
        contact_active = false;
        force = Eigen::Vector3d::Zero();
        torque = Eigen::Vector3d::Zero();
    }

    ContactData(std::string name_parent, std::string name_child, int id_parent, int id_child) {
        name_parent_geom = name_parent;
        name_child_geom = name_child;
        id_parent_geom = id_parent;
        id_child_geom = id_child;
        contact_active = false;
        force = Eigen::Vector3d::Zero();
        torque = Eigen::Vector3d::Zero();
    }

    friend std::ostream& operator<<(std::ostream& os, const ContactData& contact_data);
};

std::ostream& operator<<(std::ostream& os, const ContactData& contact_data) {
    os << "name parent: " << contact_data.name_parent_geom
        << "\tname child: " << contact_data.name_child_geom
        << "\tid parent: " << contact_data.id_parent_geom
        << "\tid child: " << contact_data.id_child_geom
        << "\tactive: " << contact_data.contact_active
        << "\tforce: " << contact_data.force.transpose()
        << "\ttorque: " << contact_data.torque.transpose();

    return os;
}

#endif