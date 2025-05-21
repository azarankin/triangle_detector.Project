import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(title, img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    plt.imshow(img, cmap='gray' if len(img.shape)==2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()


def find_shape_contour(address):
    # Load the shape template or reference image
    templat_img = cv2.imread(address)
    templat_img_gray = cv2.cvtColor(templat_img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(templat_img_gray, (5, 5), sigmaX=0)
    _, thresh1 = cv2.threshold(blured, 127, 255, 0)
    contours_template, _ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours_template, key=cv2.contourArea, reverse=True)
    if len(sorted_contours) < 2:
        raise ValueError("Template image must contain at least two contours.")

    template_img_contour = sorted_contours[1] 
    cv2.drawContours(templat_img, [template_img_contour], 0, (0, 255, 0), 3)

    return template_img_contour, templat_img

def contour_compare(target_frame, template_contour, match_threshold = 0.2, min_area_ratio=0.001, max_area_ratio=0.80):
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    #target_gray = cv2.equalizeHist(target_gray)
    blured = cv2.GaussianBlur(target_gray, (5, 5), sigmaX=0)
    thresh2 = cv2.adaptiveThreshold(blured, 255,
                                 adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                 thresholdType=cv2.THRESH_BINARY,
                                 blockSize=11,
                                 C=5)
    contours_target, _ = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_match = float('inf')
    closest_contour = []
    frame_area = target_frame.shape[0] * target_frame.shape[1]
    min_area = frame_area * min_area_ratio
    max_area = frame_area * max_area_ratio
    for c in contours_target:
        match = cv2.matchShapes(template_contour, c, 3, 0.0)
        if match < match_threshold:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue  # קטן מדי → רעש
            #min_match = match # and match < min_match
            closest_contour.append(c)

    if closest_contour:
        cv2.drawContours(target_frame, closest_contour, -1, (0, 255, 0), 3)
        
    return closest_contour, target_frame, thresh2


def run_camera(template_contour, camid = 0):
    cap = cv2.VideoCapture(camid)  # פותח את המצלמה

    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # כאן אתה תקרא ל־contour_compare בעצמך
        contour, frame, thresh2 = contour_compare(frame, template_contour)

        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def create_triangle_image(filename="triangle.png", size=(600, 600), margin = 150):
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

    pt1 = (size[0] // 2, margin)                     # top
    pt2 = (margin, size[1] - margin)                 # left
    pt3 = (size[0] - margin, size[1] - margin)       # right

    triangle_cnt = np.array([pt1, pt2, pt3])

    cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 0), -1)

    cv2.imwrite(filename, img)
    print(f"Triangle image saved as {filename}")