"""Canvas and toolbar UI for the gesture drawing application."""

from typing import List, Tuple, Optional
import cv2


class CanvasUI:
    """Maintains the drawing canvas and gesture-controlled toolbar."""

    COLORS = {
        "BLUE": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "RED": (0, 0, 255),
        "YELLOW": (0, 255, 255),
    }

    def __init__(self, brush_thickness: int = 8, header_height: int = 95):
        self.brush_thickness = brush_thickness
        self.eraser_thickness = 40
        self.header_height = header_height
        self.active_color_name = "BLUE"
        self.active_color = self.COLORS["BLUE"]
        self.canvas = None
        self._frame_width = 0
        self._frame_height = 0
        self.buttons = []

    def init_canvas(self, frame_shape: Tuple[int, int, int]) -> None:
        """Initialize or resize the drawing canvas to match the video frame."""
        height, width = frame_shape[:2]
        if self.canvas is None or self._frame_width != width or self._frame_height != height:
            self._frame_width, self._frame_height = width, height
            self.canvas = __import__('numpy').zeros((height, width, 3), dtype=__import__('numpy').uint8)
            self._create_buttons(width)

    def _create_buttons(self, width: int) -> None:
        self.buttons = []
        num_buttons = 6
        margin_x = 12
        margin_y = 10
        btn_height = self.header_height - margin_y * 2
        total_margin_space = margin_x * (num_buttons + 1)
        btn_width = (width - total_margin_space) // num_buttons
        button_specs = [
            {"id": "CLEAR", "label": "CLEAR SCREEN", "color": (50, 50, 50), "text_color": (255, 255, 255)},
            {"id": "BLUE", "label": "BLUE", "color": self.COLORS["BLUE"], "text_color": (255, 255, 255)},
            {"id": "GREEN", "label": "GREEN", "color": self.COLORS["GREEN"], "text_color": (0, 0, 0)},
            {"id": "RED", "label": "RED", "color": self.COLORS["RED"], "text_color": (255, 255, 255)},
            {"id": "YELLOW", "label": "YELLOW", "color": self.COLORS["YELLOW"], "text_color": (0, 0, 0)},
            {"id": "ERASER", "label": "ERASER", "color": (80, 80, 90), "text_color": (255, 255, 255)},
        ]
        for i, spec in enumerate(button_specs):
            x1 = margin_x + i * (btn_width + margin_x)
            y1 = margin_y
            self.buttons.append({
                "id": spec["id"], "label": spec["label"],
                "bbox": (x1, y1, x1 + btn_width, y1 + btn_height),
                "bg_color": spec["color"], "text_color": spec["text_color"],
            })

    def handle_selection(self, x: int, y: int) -> Optional[str]:
        if y > self.header_height:
            return None
        for btn in self.buttons:
            x1, y1, x2, y2 = btn["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                btn_id = btn["id"]
                if btn_id == "CLEAR":
                    self.clear_canvas()
                elif btn_id == "ERASER":
                    self.active_color_name = "ERASER"
                    self.active_color = (0, 0, 0)
                elif btn_id in self.COLORS:
                    self.active_color_name = btn_id
                    self.active_color = self.COLORS[btn_id]
                return btn_id
        return None

    def draw_stroke(self, prev_point: Optional[Tuple[int, int]], curr_point: Tuple[int, int]) -> None:
        if self.canvas is None or curr_point[1] <= self.header_height:
            return
        is_eraser = self.active_color_name == "ERASER"
        thickness = self.eraser_thickness if is_eraser else self.brush_thickness
        stroke_color = (0, 0, 0) if is_eraser else self.active_color
        if prev_point is None:
            cv2.circle(self.canvas, curr_point, thickness // 2, stroke_color, cv2.FILLED)
        elif prev_point[1] > self.header_height:
            cv2.line(self.canvas, prev_point, curr_point, stroke_color, thickness, cv2.LINE_AA)
            cv2.circle(self.canvas, curr_point, thickness // 2, stroke_color, cv2.FILLED)

    def clear_canvas(self) -> None:
        if self.canvas is not None:
            self.canvas[:] = 0

    def draw_header(self, frame: cv2.Mat) -> cv2.Mat:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._frame_width, self.header_height), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        for btn in self.buttons:
            x1, y1, x2, y2 = btn["bbox"]
            active = btn["id"] == self.active_color_name
            cv2.rectangle(frame, (x1, y1), (x2, y2), btn["bg_color"], cv2.FILLED)
            border = (255, 255, 255) if active else (80, 80, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, 4 if active else 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.55
            thickness = 2 if active else 1
            size, _ = cv2.getTextSize(btn["label"], font, scale, thickness)
            tx = x1 + (x2 - x1 - size[0]) // 2
            ty = y1 + (y2 - y1 + size[1]) // 2
            cv2.putText(frame, btn["label"], (tx, ty), font, scale,
                        btn["text_color"], thickness, cv2.LINE_AA)
        return frame

    def merge_canvas(self, frame: cv2.Mat) -> cv2.Mat:
        if self.canvas is None:
            return frame
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        return cv2.bitwise_or(frame_bg, self.canvas)
