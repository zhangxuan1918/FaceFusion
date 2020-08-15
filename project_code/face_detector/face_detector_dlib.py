import PIL
import cv2
import dlib
import numpy as np
import scipy
from project_code.face_detector.face_detector import FaceDetector


class FaceDetectorDlib(FaceDetector):

    def __init__(self, path_to_face_detector):
        self.face_detector = dlib.cnn_face_detection_model_v1(path_to_face_detector)

    def detect_from_image(self, tensor_or_path, rgb=True):
        image = self.tensor_or_path_to_ndarray(tensor_or_path, rgb=rgb)
        detected_faces = self.face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        detected_faces = [[d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()] for d in detected_faces]

        return detected_faces

    def crop_from_image(self, tensor_or_path, output_size, transform_size=4096, padding=True, rgb=True):
        if isinstance(tensor_or_path, str):
            # read image
            image = PIL.Image.open(tensor_or_path)

        detected_faces = self.detect_from_image(tensor_or_path=np.array(image), rgb=rgb)
        image_faces = []
        image_crops = []
        for face in detected_faces:
            # Calculate auxiliary vectors.
            # eye_left = np.mean(lm_eye_left, axis=0)
            # eye_right = np.mean(lm_eye_right, axis=0)
            # eye_avg = (eye_left + eye_right) * 0.5
            # eye_to_eye = eye_right - eye_left
            # mouth_left = lm_mouth_outer[0]
            # mouth_right = lm_mouth_outer[6]
            # mouth_avg = (mouth_left + mouth_right) * 0.5
            # eye_to_mouth = mouth_avg - eye_avg

            # face: [left, top, right, bottom]
            eye_to_eye = np.asarray([0.8 * (face[2] - face[0]), 0])
            eye_to_mouth = np.asarray([0, 0.65 * (face[1] - face[3])])
            eye_avg = np.asarray([0.5 * (face[2] + face[0]), 1.2 * face[1]])

            # Choose oriented crop rectangle.
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c = eye_avg + eye_to_mouth * 0.1
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
            qsize = np.hypot(*x) * 2

            # Shrink.
            shrink = int(np.floor(qsize / output_size * 0.5))
            if shrink > 1:
                rsize = (int(np.rint(float(image.size[0]) / shrink)), int(np.rint(float(image.size[1]) / shrink)))
                image = image.resize(rsize, PIL.Image.ANTIALIAS)
                quad /= shrink
                qsize /= shrink

            # Crop.
            border = max(int(np.rint(qsize * 0.1)), 3)
            crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                    int(np.ceil(max(quad[:, 1]))))
            crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, image.size[0]),
                    min(crop[3] + border, image.size[1]))
            if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
                image = image.crop(crop)
                quad -= crop[0:2]

            # Pad.
            pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                   int(np.ceil(max(quad[:, 1]))))
            pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - image.size[0] + border, 0),
                   max(pad[3] - image.size[1] + border, 0))
            if padding and max(pad) > border - 4:
                pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
                image = np.pad(np.float32(image), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                h, w, _ = image.shape
                y, x, _ = np.ogrid[:h, :w, :1]
                mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                                  1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
                blur = qsize * 0.02
                image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                image += (np.median(image, axis=(0, 1)) - image) * np.clip(mask, 0.0, 1.0)
                image = PIL.Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255)), 'RGB')
                quad += pad[:2]

            # Transform.
            image = image.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                                PIL.Image.BILINEAR)
            if output_size < transform_size:
                image = image.resize((output_size, output_size), PIL.Image.ANTIALIAS)
            image_faces.append(image)
            image_crops.append(crop)

        return image_faces, image_crops