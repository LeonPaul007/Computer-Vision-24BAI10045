"""
main.py
=======
Main controller script for the gesture drawing computer vision application.
Orchestrates webcam capture, hand tracking, UI interaction, drawing, and display.

Usage:
    python main.py
"""

import time
import cv2
from hand_tracker import HandTracker
from canvas_ui import CanvasUI


def draw_hud(
    frame: cv2.Mat,
    mode_text: str,
    active_color_name: str,
    fps: float,
) -> None:
    """Renders status indicators, mode badges, and FPS on the bottom HUD."""
    height, width, _ = frame.shape

    hud_y = height - 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, hud_y - 10), (width, height), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if mode_text == "DRAWING":
        mode_color = (0, 255, 0)
    elif mode_text == "ERASING":
        mode_color = (0, 165, 255)
    elif mode_text == "SELECTION":
        mode_color = (0, 200, 255)
    else:
        mode_color = (180, 180, 180)

    cv2.putText(frame, f"MODE: {mode_text}", (20, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"TOOL: {active_color_name}", (250, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "[C] Clear | [E] Eraser | [Q] Quit",
                (width - 440, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (width - 110, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    """Entry point for the gesture drawing application."""
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {camera_index}. Please verify camera connection.")
        return

    print("=" * 60)
    print("     GESTURE DRAWING - COMPUTER VISION APPLICATION")
    print("=" * 60)
    print(" [GUIDE] Gesture Controls:")
    print("   * Selection Mode : Raise BOTH Index & Middle fingers")
    print("     - Hover over top buttons to change color or clear canvas")
    print("   * Drawing Mode   : Raise ONLY Index finger")
    print("     - Move index finger to draw in the air")
    print("   * Pause / Standby: Lower fingers or close fist")
    print("   * Key Shortcuts  : Press 'C' to clear canvas | 'Q' or 'ESC' to quit")
    print("=" * 60)

    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.65,
    )
    canvas_ui = CanvasUI(brush_thickness=8, header_height=95)

    prev_point = None
    prev_time = time.time()
    window_name = "Gesture Drawing"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("[WARNING] Failed to grab frame from camera stream. Retrying...")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            canvas_ui.init_canvas(frame.shape)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            frame = tracker.find_hands(frame, draw=False)
            lm_list = tracker.get_landmark_positions(frame)
            mode_text = "STANDBY"

            if lm_list:
                fingers = tracker.get_fingers_up(lm_list)
                index_tip, middle_tip = tracker.get_finger_tips(lm_list)

                if index_tip and middle_tip:
                    x1, y1 = index_tip
                    x2, y2 = middle_tip
                    index_up = fingers[1]
                    middle_up = fingers[2]

                    if index_up and middle_up:
                        mode_text = "SELECTION"
                        prev_point = None
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2)
                        cv2.circle(frame, (cx, cy), 6, (0, 200, 255), cv2.FILLED)
                        canvas_ui.handle_selection(x1, y1)

                    elif index_up and not middle_up:
                        is_erasing = canvas_ui.active_color_name == "ERASER"
                        mode_text = "ERASING" if is_erasing else "DRAWING"

                        if is_erasing:
                            eraser_radius = canvas_ui.eraser_thickness // 2
                            cv2.circle(frame, (x1, y1), eraser_radius, (255, 255, 255), 2)
                            cv2.circle(frame, (x1, y1), 4, (0, 165, 255), cv2.FILLED)
                        else:
                            cv2.circle(frame, (x1, y1), canvas_ui.brush_thickness,
                                       canvas_ui.active_color, cv2.FILLED)
                            cv2.circle(frame, (x1, y1), canvas_ui.brush_thickness + 2,
                                       (255, 255, 255), 2)

                        canvas_ui.draw_stroke(prev_point, (x1, y1))
                        prev_point = (x1, y1)
                    else:
                        prev_point = None
            else:
                prev_point = None

            output_frame = canvas_ui.merge_canvas(frame)
            output_frame = canvas_ui.draw_header(output_frame)
            draw_hud(output_frame, mode_text, canvas_ui.active_color_name, fps)
            cv2.imshow(window_name, output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[INFO] Exiting application...")
                break
            elif key == ord('c') or key == ord('C'):
                canvas_ui.clear_canvas()
            elif key == ord('e') or key == ord('E'):
                canvas_ui.active_color_name = "ERASER"
                canvas_ui.active_color = (0, 0, 0)

    except KeyboardInterrupt:
        print("\n[INFO] Application interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete. Webcam and OpenCV windows closed.")


if __name__ == "__main__":
    main()
