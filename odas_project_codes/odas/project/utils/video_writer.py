"""video_writer.py
"""


import subprocess

import cv2


def get_video_writer(name, width, height, fps=30):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    filename = name + '.mp4'  # MP4
    return cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


