"""
Simple CLI runner for the reusable recognition stream.

Shows annotated frames in an OpenCV window and logs recognition events while
leaving the heavy lifting to recognition.stream. Press Q/ESC to exit.
"""

import cv2
import numpy as np

from recognition.settings import SettingsStore
from recognition.stream import RecognitionStream


def main():
    settings_store = SettingsStore()
    stream = RecognitionStream(settings_store)
    stream.start()

    print("Готово. Держите лицо в кадре 2–3 сек для срабатывания события. Выход: Q/ESC.")
    try:
        for frame_bytes in stream.frames():
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            cv2.imshow("Safeschool - Recognition", frame)

            # Print pending status/event messages without blocking
            while True:
                msg = stream.messages(timeout=0.01)
                if not msg:
                    break
                if msg.get("type") == "event":
                    data = msg.get("data", {})
                    print(f"{data.get('text')}")
                elif msg.get("type") == "status":
                    data = msg.get("data", {})
                    print(f"[status] {data}")

            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
