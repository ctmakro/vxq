# MSMF hack
# https://github.com/opencv/opencv/issues/17687
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
