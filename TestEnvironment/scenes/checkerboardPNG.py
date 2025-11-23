import cv2
import numpy as np

squares_x = 9
squares_y = 9
square_px = 100  # Pixel per square

img = np.zeros((squares_y*square_px, squares_x*square_px), np.uint8)

for y in range(squares_y):
    for x in range(squares_x):
        if (x + y) % 2 == 0:
            cv2.rectangle(img,
                (x*square_px, y*square_px),
                ((x+1)*square_px, (y+1)*square_px),
                255,
                -1
            )

cv2.imwrite("checker9x9.png", img)
