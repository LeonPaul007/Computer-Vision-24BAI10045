"""
hand_tracker.py
================
Module responsible for hand detection and landmark tracking using MediaPipe.
"""

from typing import List, Tuple, Optional
import cv2
import mediapipe as mp


class HandTracker:
    """Tracks hand landmarks and evaluates finger states."""

    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [3, 6, 10, 14, 18]

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, frame: cv2.Mat, draw: bool = True) -> cv2.Mat:
        """Detect hands in a BGR frame and optionally draw landmarks."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def get_landmark_positions(self, frame: cv2.Mat) -> List[Tuple[int, int]]:
        """Return pixel coordinates for the first detected hand."""
        landmarks: List[Tuple[int, int]] = []
        if not self.results or not self.results.multi_hand_landmarks:
            return landmarks

        height, width, _ = frame.shape
        hand = self.results.multi_hand_landmarks[0]
        for landmark in hand.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height)))
        return landmarks

    def get_fingers_up(self, lm_list: List[Tuple[int, int]]) -> List[bool]:
        """Estimate which fingers are raised using landmark geometry."""
        if len(lm_list) < 21:
            return [False] * 5

        fingers = [False] * 5
        fingers[0] = lm_list[4][0] < lm_list[3][0]
        for i, (tip, pip) in enumerate(zip(self.TIP_IDS[1:], self.PIP_IDS[1:]), start=1):
            fingers[i] = lm_list[tip][1] < lm_list[pip][1]
        return fingers

    def get_finger_tips(
        self, lm_list: List[Tuple[int, int]]
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Return index and middle fingertip coordinates."""
        if len(lm_list) < 21:
            return None, None
        return lm_list[8], lm_list[12]

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
