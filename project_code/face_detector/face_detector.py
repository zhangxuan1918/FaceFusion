
from abc import ABC

import PIL
import cv2
import dlib
import tensorflow as tf
import numpy as np
from skimage import io

"""
copied from https://github.com/1adrianb/face-alignment
"""


class FaceDetector(ABC):

    def detect_from_image(self, tensor_or_path):

        """Detects faces in a given image.

        This function detects the faces present in a provided BGR(usually)
        image. The input can be either the image itself or the path to it.

        Arguments:
            tensor_or_path {numpy.ndarray, torch.tensor or string} -- the path
            to an image or the image itself.

        Example::

            >>> path_to_image = 'data/image_01.jpg'
            ...   detected_faces = detect_from_image(path_to_image)
            [A list of bounding boxes (x1, y1, x2, y2)]
            >>> image = cv2.imread(path_to_image)
            ...   detected_faces = detect_from_image(image)
            [A list of bounding boxes (x1, y1, x2, y2)]

        """
        raise NotImplementedError

    @staticmethod
    def tensor_or_path_to_ndarray(tensor_or_path, rgb=True):
        """Convert path (represented as a string) or torch.tensor to a numpy.ndarray

        Arguments:
            tensor_or_path {numpy.ndarray, torch.tensor or string} -- path to the image, or the image itself
        """
        if isinstance(tensor_or_path, str):
            return cv2.imread(tensor_or_path) if not rgb else io.imread(tensor_or_path)
        elif tf.is_tensor(tensor_or_path):
            # Call cpu in case its coming from cuda
            return tensor_or_path.cpu().numpy()[..., ::-1].copy() if not rgb else tensor_or_path.cpu().numpy()
        elif isinstance(tensor_or_path, np.ndarray):
            return tensor_or_path[..., ::-1].copy() if not rgb else tensor_or_path
        else:
            raise TypeError

    # def __align_face(self, image, lm, output_size, transform_size=512, enable_padding=True):
    #     lm_eye_left = lm[36: 42]  # left-clockwise
    #     lm_eye_right = lm[42: 48]  # left-clockwise
    #     lm_mouth_outer = lm[48: 60]  # left-clockwise
    #
    #     # Calculate auxiliary vectors.
    #     eye_left = np.mean(lm_eye_left, axis=0)
    #     eye_right = np.mean(lm_eye_right, axis=0)
    #     eye_avg = (eye_left + eye_right) * 0.5
    #     eye_to_eye = eye_right - eye_left
    #     mouth_left = lm_mouth_outer[0]
    #     mouth_right = lm_mouth_outer[6]
    #     mouth_avg = (mouth_left + mouth_right) * 0.5
    #     eye_to_mouth = mouth_avg - eye_avg
    #
    #     # Choose oriented crop rectangle.
    #     x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    #     x /= np.hypot(*x)
    #     x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    #     y = np.flipud(x) * [-1, 1]
    #     c = eye_avg + eye_to_mouth * 0.1
    #     quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    #     qsize = np.hypot(*x) * 2
    #
    #     # Shrink.
    #     shrink = int(np.floor(qsize / output_size * 0.5))
    #     if shrink > 1:
    #         rsize = (int(np.rint(float(image.size[0]) / shrink)), int(np.rint(float(image.size[1]) / shrink)))
    #         image = image.resize(rsize, PIL.Image.ANTIALIAS)
    #         quad /= shrink
    #         qsize /= shrink
    #
    #     # Crop.
    #     border = max(int(np.rint(qsize * 0.1)), 3)
    #     crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
    #             int(np.ceil(max(quad[:, 1]))))
    #     crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, image.size[0]),
    #             min(crop[3] + border, image.size[1]))
    #     if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
    #         image = image.crop(crop)
    #         quad -= crop[0:2]
    #
    #     # Pad.
    #     pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
    #            int(np.ceil(max(quad[:, 1]))))
    #     pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - image.size[0] + border, 0),
    #            max(pad[3] - image.size[1] + border, 0))
    #     if enable_padding and max(pad) > border - 4:
    #         pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
    #         image = np.pad(np.float32(image), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
    #         h, w, _ = image.shape
    #         y, x, _ = np.ogrid[:h, :w, :1]
    #         mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
    #                           1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
    #         blur = qsize * 0.02
    #         image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(mask * 3.0 + 1.0, 0.0,
    #                                                                                            1.0)
    #         image += (np.median(image, axis=(0, 1)) - image) * np.clip(mask, 0.0, 1.0)
    #         image = PIL.Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255)), 'RGB')
    #         quad += pad[:2]
    #
    #     # Transform.
    #     image = image.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
    #                             PIL.Image.BILINEAR)
    #     if output_size < transform_size:
    #         image = image.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    #
    #     # Save aligned image.
    #     return image, crop
    #
    # def align_face_from_image(self, tensor_or_path, lm_detector, aligned_size, transform_size=512, lms=None, enable_padding=True):
    #     """
    #     align face in the image, the alignment is compatible with dataset FFHQ
    #     :param tensor_or_path:
    #     :return:
    #     """
    #     image = self.tensor_or_path_to_ndarray(tensor_or_path)
    #     if lms is None:
    #         lms = []
    #         detected_faces = self.detect_from_image(tensor_or_path=image)
    #
    #         for face in detected_faces:
    #             lm_tmp = lm_detector(image, dlib.rectangle(*face))
    #             lmp_tmp = list(lm_tmp.parts())
    #             lmp = []
    #             for p in lmp_tmp:
    #                 lmp.append([p.x, p.y])
    #             lms.append(lmp)
    #         # lms shape (n, 68, 2)
    #         lms = np.asarray(lms)
    #
    #     image = PIL.Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255)), 'RGB')
    #     aligned_face = []
    #     for lm in lms:
    #         img_cropped, crop = self.__align_face(
    #             image=image,
    #             lm=lm,
    #             output_size=aligned_size,
    #             transform_size=transform_size,
    #             enable_padding=enable_padding)
    #         aligned_face.append((img_cropped, crop))
    #
    #     return aligned_face