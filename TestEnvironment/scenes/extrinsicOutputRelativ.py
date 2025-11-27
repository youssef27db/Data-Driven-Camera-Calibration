import bpy
import json
import mathutils

# Blender -> OpenCV Koordinatentransformation
B_TO_CV = mathutils.Matrix((
    (1,  0,  0),
    (0, -1,  0),
    (0,  0, -1),
))

# Mapping: so heißen deine Kamera-Objekte in Blender
CAM_OBJECTS = {
    "Center": "Camera_Center",
    "Up1":    "Camera_Up1",
    "Up2":    "Camera_Up2",
    "Up3":    "Camera_Up3",
    "Down1":  "Camera_Down1",
    "Down2":  "Camera_Down2",
    "Down3":  "Camera_Down3",
    "Left1":  "Camera_Left1",
    "Left2":  "Camera_Left2",
    "Left3":  "Camera_Left3",
    "Right1": "Camera_Right1",
    "Right2": "Camera_Right2",
    "Right3": "Camera_Right3",
}

def get_pose_cv(obj: bpy.types.Object):
    """Pose einer Kamera im OpenCV-Koordinatensystem (Weltpose)."""
    M = obj.matrix_world.copy()
    loc, rot_quat, scale = M.decompose()

    # reine Rotation ohne Scale
    R_bl = rot_quat.to_matrix().to_3x3()
    t_bl = loc

    # nach OpenCV umrechnen
    R_cv = B_TO_CV @ R_bl @ B_TO_CV.transposed()
    t_cv = B_TO_CV @ t_bl

    return R_cv, t_cv


def export_relative_extrinsics(save_path: str):
    # Referenzkamera = Center
    ref_name = "Center"
    ref_obj = bpy.data.objects[CAM_OBJECTS[ref_name]]

    R0, t0 = get_pose_cv(ref_obj)
    R0_T = R0.transposed()

    output = {}

    for cam_id, obj_name in CAM_OBJECTS.items():
        cam_obj = bpy.data.objects[obj_name]
        Rj, tj = get_pose_cv(cam_obj)

        if cam_id == "Center":
            R_rel = mathutils.Matrix.Identity(3)
            T_rel = mathutils.Vector((0.0, 0.0, 0.0))
        else:
            # relative Rot/Trans zu Center
            R_rel = Rj @ R0_T
            T_rel = tj - R_rel @ t0

        R_list = [[float(R_rel[i][j]) for j in range(3)] for i in range(3)]
        T_list = [float(T_rel.x), float(T_rel.y), float(T_rel.z)]

        output[cam_id] = {
            "rotationMatrix": R_list,
            "translationVector": T_list
        }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print("Export complete →", save_path)


# Ausführen:
save_path = bpy.path.abspath("//groundtruth_extrinsics_relative.json")
export_relative_extrinsics(save_path)
