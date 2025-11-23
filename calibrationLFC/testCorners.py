import cv2
from InitialCalibration import InitialCalibration

# 9x9 Squares -> 8x8 inner corners
CAL = InitialCalibration(chessboardSize=(8, 8), squareSize=0.02)

def debug_image(path):
    print("Testing:", path)
    ok, corners, _ = CAL.detectCorners(path)
    print("  found:", ok)

    img = cv2.imread(path)
    if img is None:
        print("  ERROR: could not load image")
        return

    # Wenn erkannt → Ecken draufmalen
    if ok:
        cv2.drawChessboardCorners(img, CAL.chessboardSize, corners, ok)

    cv2.imshow(path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug_image("imageset/pose_000_Center.png")
    debug_image("imageset/pose_000_Right1.png")
    debug_image("imageset/pose_000_Right3.png")
