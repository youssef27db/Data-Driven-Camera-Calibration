"""
Small test runner for initial calibration.

How to run (from repository root):
  cd calibrationLFC
  python run_initial_calibration.py

This script performs quick checks (file existence, corner detection on pose 0)
and then calls the controller to run the initial calibration. It prints a short
summary at the end.
"""

import os
import sys

from ImageSet import ImageSet
from Controller import Controller


# -------------------------
# Configuration
# -------------------------
BASE_DIR = "imageset"  # folder with images
NUM_POSES = 1
CAM_IDS = [
    "Center",
    "Up1", "Up2", "Up3",
    "Down1", "Down2", "Down3",
    "Left1", "Left2", "Left3",
    "Right1", "Right2", "Right3",
]


def main():
    print("Creating ImageSet...")
    imageSet = ImageSet(BASE_DIR, NUM_POSES, CAM_IDS)

    print("Instantiating Controller and InitialCalibration...")
    controller = Controller()

    print("\nQuick checks (pose 0): file exists / corners detected")
    for cam in CAM_IDS:
        path = imageSet.getImagePath(0, cam)
        exists = os.path.exists(path)
        cornersFound = None

        try:
            cornersFound, corners, size = controller.initialCalibration.detectCorners(path)
        except Exception as e:
            cornersFound = f"error: {e.__class__.__name__}"
        print(f" {cam}: {path} - exists={exists} - corners={cornersFound}")

    print("\nRunning initial calibration (this may take time depending on data)...")
    try:
        calibrationState = controller.runInitialCalibration(imageSet)
    except Exception as e:
        print("Calibration failed with exception:", e)
        # re-raise so CI or interactive runs show the traceback
        raise

    print("\nCalibration finished. Attempting to print summary...")
    try:
        stateDict = calibrationState.__getState__()
    except Exception:
        stateDict = None

    if stateDict is not None:
        intr = stateDict.get("intrinsics", {})
        extr = stateDict.get("extrinsics", {})
        print(f" - Intrinsics available for cameras: {list(intr.keys())}")
        print(f" - Extrinsics available for cameras: {list(extr.keys())}")
    else:
        print("Could not extract a state dictionary. The returned object:", calibrationState)



if __name__ == "__main__":
    # If the script is executed from repository root, adjust sys.path so imports work
    repoRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repoRoot not in sys.path:
        sys.path.insert(0, repoRoot)
    main()
