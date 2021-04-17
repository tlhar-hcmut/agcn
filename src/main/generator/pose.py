import cv2
import numpy as np
from mediapipe.python.solutions import drawing_utils, pose

from . import processor


def draw_pose_img(filename: str):
    with pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as extracter:
        image = cv2.imread(filename)
        results = extracter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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


def _get_skeleton(result):
    def _get_joint(idx: int):
        return np.array([result[idx].z, result[idx].x, result[idx].y])

    hip_center = (_get_joint(23) + _get_joint(24)) / 2
    shoulder_center = (_get_joint(9) + _get_joint(10)) / 2
    mouth = (_get_joint(9) + _get_joint(10)) / 2
    return np.array(
        [
            hip_center,  # 1 hip_center
            (hip_center + shoulder_center) / 2,  # 2 spine_center
            (mouth + shoulder_center) / 2,  # 3 spine_center
            _get_joint(0),  # 4 head
            _get_joint(11),  # 5  left_shoulder
            _get_joint(13),  # 6  left_elbow
            _get_joint(15),  # 7  left_wrist
            _get_joint(15),  # 8  left_hand # TODO
            _get_joint(12),  # 9  right_shoulder
            _get_joint(14),  # 10 right_elbow
            _get_joint(16),  # 11 right_wrist
            _get_joint(16),  # 12 right_hand # TODO
            _get_joint(23),  # 13 left_hip
            _get_joint(25),  # 14 left_knee
            _get_joint(27),  # 15 left_ankle
            _get_joint(31),  # 16 left_foot
            _get_joint(24),  # 17 right_hip
            _get_joint(26),  # 18 right_knee
            _get_joint(28),  # 19 right_ankle
            _get_joint(30),  # 20 right_foot
            shoulder_center,  # 21 center_shoulder
            (_get_joint(19) + _get_joint(17)) / 2,  # 22 left_hand_tip
            _get_joint(21),  # 23 left_hand_thump
            (_get_joint(18) + _get_joint(20)) / 2,  # 24 right_hand_tip
            _get_joint(22),  # 25 right_hand_thump
        ]
    )


def get_skeleton_by_frame(filename: str):
    cap = cv2.VideoCapture(filename)
    with pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as extracter:
        ls_frame = []
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            result = extracter.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ).pose_landmarks.landmark
            ls_frame.append(_get_skeleton(result))
    cap.release()

    output = np.expand_dims(np.array(ls_frame).transpose(2, 0, 1), axis=[-1, 0])

    return np.array(np.squeeze(output, axis=0))


if __name__ == "__main__":
    print(get_skeleton_by_frame("output/pose/shaking-hands.mp4").shape)
