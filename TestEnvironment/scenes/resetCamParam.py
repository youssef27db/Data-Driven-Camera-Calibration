import bpy
import math

# Rotation in Radiant
rot_x = math.radians(90)
rot_y = math.radians(0)
rot_z = math.radians(90)

# Hilfsfunktion: Kamera holen oder erzeugen
def get_or_create_camera(name):
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    else:
        cam_data = bpy.data.cameras.new(name)
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        return cam_obj

# Kamera-Positionen definieren
camera_positions = {
    "Camera_Center": (0, 0, 2),

    "Camera_Down1": (0, 0, 1.5),
    "Camera_Down2": (0, 0, 1.0),
    "Camera_Down3": (0, 0, 0.5),

    "Camera_Up1": (0, 0, 2.5),
    "Camera_Up2": (0, 0, 3.0),
    "Camera_Up3": (0, 0, 3.5),

    "Camera_Left1": (0, -0.5, 2),
    "Camera_Left2": (0, -1.0, 2),
    "Camera_Left3": (0, -1.5, 2),

    "Camera_Right1": (0, 0.5, 2),
    "Camera_Right2": (0, 1.0, 2),
    "Camera_Right3": (0, 1.5, 2),
}

# Kameras erzeugen / setzen
for name, pos in camera_positions.items():
    cam = get_or_create_camera(name)
    cam.location = pos
    cam.rotation_euler = (rot_x, rot_y, rot_z)

print("Alle Kameras wurden erstellt und gesetzt.")
