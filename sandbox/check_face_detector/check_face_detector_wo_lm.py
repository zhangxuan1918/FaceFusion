import cv2

from project_code.face_detector.face_detector_dlib import FaceDetectorDlib


path_to_detector = '/opt/data/dlib/mmod_human_face_detector.dat'
face_detector = FaceDetectorDlib(path_to_face_detector=path_to_detector)
img_path = '/opt/project/input/images/random/pic16.jpeg'
output_path = './bb.jpg'
croped_image, crops = face_detector.crop_from_image(img_path, 224)
img = cv2.imread(img_path)

# for rect in crops:
#
#     x = rect[0]
#     y = rect[1]
#     w = rect[2] - x
#     h = rect[3] - y
#
#     wh = max(w, h)
#     cv2.rectangle(img, (x, y), (x + wh, y + wh), (255, 255, 0), 2)
#
# cv2.imwrite(output_path, img)
croped_image[0].save(output_path)