import bpy
import json
import mathutils

B_TO_CV = mathutils.Matrix((
    (1,  0,  0),
    (0, -1,  0),
    (0,  0, -1),
))

output = {}

for obj in bpy.data.objects:
    if obj.type == 'CAMERA':

        name = obj.name

        # Welt-Transform holen
        world_matrix = obj.matrix_world.copy()

        T_bl = world_matrix.to_translation()
        R_bl = world_matrix.to_3x3()

        # in OpenCV-Koordinatensystem umrechnen
        T_cv = B_TO_CV @ T_bl         
        R_cv = B_TO_CV @ R_bl          
        
        # In normale Python-Listen umwandeln
        T_list = [float(T_cv.x), float(T_cv.y), float(T_cv.z)]
        R_list = [[float(R_cv[i][j]) for j in range(3)] for i in range(3)]

        output[name] = {
            "rotationMatrix": R_list,
            "translationVector": T_list
        }

# JSON speichern
save_path = bpy.path.abspath("//groundtruth_extrinsics_misaligned.json")
with open(save_path, 'w') as f:
    json.dump(output, f, indent=2)

print("Export complete →", save_path)
