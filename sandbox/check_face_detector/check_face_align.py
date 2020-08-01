import dlib

from project_code.face_detector.face_detector_dlib import FaceDetectorDlib
import cv2


path_to_detector = '/opt/data/dlib/mmod_human_face_detector.dat'
face_detector = FaceDetectorDlib(path_to_face_detector=path_to_detector)
lm_detector = dlib.shape_predictor('/opt/data/dlib/shape_predictor_68_face_landmarks.dat')

img_path = '/opt/project/input/images/random/pic9.jpeg'
output_path = './bb.jpg'
img = cv2.imread(img_path)

res = face_detector.align_face_from_image(
    tensor_or_path=img,
    lm_detector=lm_detector,
    aligned_size=224)


for _, crop in res:
    x = crop[0]
    y = crop[1]
    w = crop[2] - x
    h = crop[3] - y

    wh = max(w, h)
    cv2.rectangle(img, (x, y), (x + wh, y + wh), (255, 255, 0), 2)

cv2.imwrite(output_path, img)