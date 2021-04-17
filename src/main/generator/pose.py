from typing import NamedTuple

import cv2
from mediapipe.python.solutions import drawing_utils, pose


def draw_pose_img(filename: str):
    with pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as extracter:
        image = cv2.imread(filename)
        results: NamedTuple = extracter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        img_annot = image.copy()
        drawing_utils.draw_landmarks(
            image=img_annot,
            landmark_list=results.pose_landmarks,
            connections=pose.POSE_CONNECTIONS,
        )

        cv2.imwrite(filename.split(".")[0] + "-pose" + ".png", img_annot)


def draw_pose_video(filename=0):
    cap = cv2.VideoCapture(filename)
    with pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as extracter:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = extracter.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            drawing_utils.draw_landmarks(
                image=image,
                landmark_list=results.pose_landmarks,
                connections=pose.POSE_CONNECTIONS,
            )
            cv2.imshow("MediaPipe Pose", image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

