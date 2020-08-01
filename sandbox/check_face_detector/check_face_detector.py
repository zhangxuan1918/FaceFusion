from project_code.face_detector.face_detector_dlib import FaceDetectorDlib
import cv2


path_to_detector = '/opt/data/dlib/mmod_human_face_detector.dat'
face_detector = FaceDetectorDlib(path_to_face_detector=path_to_detector)
img_path = '/opt/project/input/images/random/pic16.jpeg'
output_path = './bb.jpg'
img = cv2.imread(img_path)
rects = face_detector.detect_from_image(img_path)

for (i, rect) in enumerate(rects):

    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y

    x = int(x / 1.5)
    y = int(y / 1.5)
    w = int(w * 1.5)
    h = int(h * 1.5)

    wh = max(w, h)
    cv2.rectangle(img, (x, y), (x + wh, y + wh), (255, 255, 0), 2)

cv2.imwrite(output_path, img)