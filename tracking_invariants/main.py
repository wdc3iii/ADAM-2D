import yaml
from plot.logger import Logger
from tracking_invariants.tracking_invariant import TrackingInvariant

mesh_path = "rsc/models/"
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
    approxMethod = config["approxMethod"]
    vis = config["vis"]

    tracking_invariant = TrackingInvariant(v_ref, z_ref, pitch_ref, T_SSP, approxMethod, 1000, Njobs=5, use_static_com=use_static_com, useAngMomState=useAngMomState, gravity_comp=gravity_comp, use_task_space_ctrl=use_task_space_ctrl, visualize=vis)
    
    while tracking_invariant.getPropInSet() < 1:
        tracking_invariant.iterateSetMap(verbose=True)

    # print("q0: ", q0, "\nqd0: ", qd0, "qC0: ", qC0, "\nqdC0: ", qdC0, "\n\nqf: ", qF, "\nqdf: ", qdF, "qCF: ", qCF, "\nqdCF: ", qdCF)



if __name__ == "__main__":
    main()