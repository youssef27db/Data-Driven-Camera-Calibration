import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Dieser Dateiordner = calibrationLFC/plots
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# calibrationLFC-Ordner
CALIB_LFC_DIR = os.path.dirname(THIS_DIR)

# Projektroot = eine Ebene höher (CameraCalibration)
PROJECT_ROOT = os.path.dirname(CALIB_LFC_DIR)

CALIB_PATH = os.path.join(
    CALIB_LFC_DIR, "results",
    "calibration_initial_imageset2_20251201_214619.json" 
)

GT_PATH = os.path.join(
    PROJECT_ROOT, "TestEnvironment", "params",
    "groundtruth_extrinsics_relative.json"    
)

CAM_IDS = [
    "Center",
    "Up1", "Up2", "Up3",
    "Down1", "Down2", "Down3",
    "Left1", "Left2", "Left3",
    "Right1", "Right2", "Right3",
]

def rot_angle_deg(R):
    """Berechnet den Rotationswinkel aus einer Rotationsmatrix."""
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def main():
    # Kalibrierung laden
    with open(CALIB_PATH, "r") as f:
        calib = json.load(f)
    extr_est = calib["state"]["extrinsics"]

    # Groundtruth laden
    with open(GT_PATH, "r") as f:
        gt = json.load(f)

    rot_err = []
    trans_err = []

    for cam in CAM_IDS:
        R_est = np.array(extr_est[cam]["rotationMatrix"], dtype=float)
        T_est = np.array(extr_est[cam]["translationVector"], dtype=float).reshape(3)

        R_gt = np.array(gt[cam]["rotationMatrix"], dtype=float)
        T_gt = np.array(gt[cam]["translationVector"], dtype=float).reshape(3)

        # Rotationsfehler
        R_err = R_est @ R_gt.T
        ang = rot_angle_deg(R_err)

        # Translationsfehler
        dT = np.linalg.norm(T_est - T_gt)

        rot_err.append(ang)
        trans_err.append(dT)

    x = np.arange(len(CAM_IDS))

    plt.figure(figsize=(14, 6))

    # ------------------- Rotationsfehler -------------------
    plt.subplot(1, 2, 1)
    plt.plot(x, rot_err, marker="o", linestyle="-", linewidth=2, color="#7f3cff")
    plt.xticks(x, CAM_IDS, rotation=45)
    plt.ylabel("Rotationsfehler (°)")
    plt.title("Rotationsfehler pro Kamera")
    plt.grid(alpha=0.3)

    # ------------------- Translationsfehler -------------------
    plt.subplot(1, 2, 2)
    plt.plot(x, trans_err, marker="s", linestyle="-", linewidth=2, color="#ff3ccf")
    plt.xticks(x, CAM_IDS, rotation=45)
    plt.ylabel("Translationsfehler (m)")
    plt.title("Translationsfehler pro Kamera")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()