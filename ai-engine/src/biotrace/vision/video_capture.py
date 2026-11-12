"""Open video streams with a bounded FFmpeg decoder thread pool."""

import cv2


def open_video_capture(path: str) -> cv2.VideoCapture:
    """Open a stream with bounded buffers and explicit display rotation.

    FFmpeg auto-applies rotation metadata when supported. We request
    ``CAP_PROP_ORIENTATION_AUTO`` explicitly so portrait phone clips are
    decoded in display orientation across backends. We never apply a second
    manual rotation.
    """
    thread_property = getattr(cv2, "CAP_PROP_N_THREADS", None)
    auto_orientation = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    capture = None

    if thread_property is not None:
        capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [thread_property, 4])
        if not capture.isOpened():
            capture.release()
            capture = None

    if capture is None:
        capture = cv2.VideoCapture(path)

    if capture.isOpened() and auto_orientation is not None:
        try:
            capture.set(auto_orientation, 1.0)
        except Exception:
            pass
    return capture
