import glob
import os
from pathlib import Path
import subprocess
from fractions import Fraction
import dlib
import numpy as np
import re
import argparse
import cv2
import shutil

import numpy as np
import cv2
from collections import OrderedDict

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


# For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
    ("right_eye", (2, 3)),
    ("left_eye", (0, 1)),
    ("nose", (4))
])


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def resize(image, width=None, height=None, inter=cv2.INTER_CUBIC):
    # initialize the dimensions of the reference_image to be resized and
    # grab the reference_image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original reference_image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the reference_image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized reference_image
    return resized


class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, reference_image, reference_gray, rect, compressed_image=None, landmarks=None):
        # convert the landmark (x, y)-coordinates to a NumPy array
        if landmarks is not None:
            shape = landmarks
        else:
            shape = self.predictor(reference_gray, rect)
            shape = shape_to_np(shape)

        # simple hack ;)
        if (len(shape) == 68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting reference_image by taking
        # the ratio of the distance between eyes in the *current*
        # reference_image to the ratio of distance between eyes in the
        # *desired* reference_image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input reference_image
        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
                      int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        landmarks = cv2.transform(np.array([shape]), M)[0]

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        reference_output = cv2.warpAffine(reference_image, M, (w, h),
                                flags=cv2.INTER_CUBIC)
        if compressed_image is not None:
            compressed_output = cv2.warpAffine(compressed_image, M, (w, h),
                                          flags=cv2.INTER_CUBIC)

            # return the aligned face
            return reference_output, compressed_output, landmarks, M
        return reference_output, landmarks, M


def compress_videos(path, crf):
    Path(f"{path}/compressed_{crf}").mkdir(parents=True, exist_ok=True)
    file_name = 'failure_project.mp4'
    os.system(f"ffmpeg -i {path}/original/{file_name} -c:v libx264 -crf {crf} -an {path}/compressed_{crf}/{file_name}")


def extract_inference_frames(base_path):
    
    file = f"{base_path}/failure_project.mp4"
    file_name = 'failure_project.mp4'
    # print(f"{file_name}")
    path_frames = f"{base_path}/{file_name[:-4]}/frames"
    key_frames = f"{path_frames}_key"
    Path(path_frames).mkdir(parents=True, exist_ok=True)
    Path(key_frames).mkdir(parents=True, exist_ok=True)
    output_fps = subprocess.check_output(f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {file}", shell=True)
    output_fps = output_fps.decode("utf-8").strip("\n")
    fps = round(Fraction(output_fps))
    os.system(f"ffmpeg -i {file} -qscale:v 2 {path_frames}/%00d.jpg")
    num_frames = len(glob.glob(f"{path_frames}/*.jpg"))
    print({path_frames})
    for idx in range(1, num_frames, fps):
        shutil.copy2(f"{path_frames}/{idx}.jpg", f"{key_frames}/{idx}_key.jpg")
        # os.system(f"copy {path_frames}/{idx}.jpg {key_frames}/{idx}_key.jpg")
        # subprocess.run(["move", f'{path_frames}/{idx}.jpg', f'{path_frames}/{idx}_key.jpg"'], check=True, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        


def crop_and_align(base_path, crf):
    path_compressed = f"{base_path}/compressed_{crf}"
    path_original = f"{base_path}/original"

    # face_detector = dlib.cnn_face_detection_model_v1("pretrained_models/dlib_weights.dat")
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor("pretrained_models/dlib_shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner(landmark_detector, desiredLeftEye=(0.38, 0.45), desiredFaceWidth=256)

    seconds_limit = 30  # Stop cropping after seconds_limit of video (i.e. seconds_limit*fps frames)

    
    video_name = 'failure_project'

    Path(f"{path_compressed}/{video_name}/crops").mkdir(parents=True, exist_ok=True)
    Path(f"{path_compressed}/{video_name}/binary_landmarks").mkdir(parents=True, exist_ok=True)
    Path(f"{path_compressed}/{video_name}/landmarks").mkdir(parents=True, exist_ok=True)
    Path(f"{path_compressed}/{video_name}/transform_matrices").mkdir(parents=True, exist_ok=True)
    Path(f"{path_original}/{video_name}/crops").mkdir(parents=True, exist_ok=True)
    
    p = (f"{path_compressed}/{video_name}/frames_key")
    # print(f"{path_compressed}/{video_name}/frames")
    keyframes = sorted([int(el[:-8]) for el in os.listdir(p) if "key" in el])    # Take only keyframes and remove "_key.jpg" to cast to int
    fps = keyframes[1] - keyframes[0]

    last_rect = None
    last_landmarks = None

    frames = os.listdir("./data/compressed_42/failure_project/frames")
    frames.sort(key=lambda f: int(re.sub('\D', '', f)))  # Sort frames correctly to crop the first 30 seconds of each video
    frames = [f"./data/compressed_42/failure_project/frames/{frame}" for frame in frames]
    for i, frame in enumerate(frames):
        frame_name = os.path.basename(frame)

        frame_compressed = cv2.imread(frame)
        frame_original = cv2.imread(f"./data/original/failure_project/frames/{frame_name}")
        gray_frame_compressed = cv2.cvtColor(frame_compressed, cv2.COLOR_BGR2GRAY)

        face_compressed = face_detector(gray_frame_compressed, 1)
        transform_landmarks = False
        
        if len(face_compressed) != 0:
            last_rect = face_compressed[0]
            landmarks = landmark_detector(gray_frame_compressed, last_rect)
            if landmarks is not None:
                last_landmarks = shape_to_np(landmarks)
                transform_landmarks = True
        crop_compressed, crop_original, transformed_landmarks, transform_matrix = face_aligner.align(frame_compressed, gray_frame_compressed,
                                                                                                        rect=last_rect,
                                                                                                        compressed_image=frame_original,
                                                                                                        landmarks=last_landmarks)
        if transform_landmarks:
            last_landmarks = transformed_landmarks

        landmark_img = generate_landmark_binary_image(crop_compressed.shape, last_landmarks)

        cv2.imwrite(f"{path_compressed}/{video_name}/crops/{frame_name}", crop_compressed)
        cv2.imwrite(f"{path_compressed}/{video_name}/binary_landmarks/{frame_name}", landmark_img)
        np.save(f"{path_compressed}/{video_name}/landmarks/{frame_name[:-4]}.npy", last_landmarks)
        np.save(f"{path_compressed}/{video_name}/transform_matrices/{frame_name[:-4]}.npy", transform_matrix)

        cv2.imwrite(f"{path_original}/{video_name}/crops/{frame_name}", crop_original)

        if i >= fps * seconds_limit:
            break


def generate_landmark_binary_image(frame_shape, landmarks):
    frame_height, frame_width = frame_shape[:2]
    binary_image = np.zeros((frame_height, frame_width, 1))
    for i in range(landmarks.shape[0]):
        landmark_x = min(max(landmarks[i][0], 0), frame_height - 1)
        landmark_y = min(max(landmarks[i][1], 0), frame_width - 1)
        binary_image[landmark_y, landmark_x, 0] = 255

    return binary_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="HQ videos should be in {BASE_PATH}/original")
    parser.add_argument("--crf", type=int, default=42, help="Constant Rate Factor")
    args = parser.parse_args()

    BASE_PATH = args.base_path  # HQ videos should be in {BASE_PATH}/original
    CRF = args.crf
    compress_videos(BASE_PATH, CRF)
    extract_inference_frames(f"{BASE_PATH}/original")
    extract_inference_frames(f"{BASE_PATH}/compressed_{CRF}")
    crop_and_align(BASE_PATH, CRF)