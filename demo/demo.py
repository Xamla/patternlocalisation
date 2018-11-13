#!/usr/bin/env python

import numpy as np
import sys
import cv2 as cv
import os
import math
from matplotlib import pyplot as plt

from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import eig

from copy import copy, deepcopy

from patternlocalisation.pattern_localisation import PatternLocalisation


# Load stereo calibration
calib_fn = "stereo_cams_4103130811_4103189394.npy"
stereoCalib = np.load(calib_fn).item()

# Create pattern localizer
pattern_localizer = PatternLocalisation()
pattern_localizer.circleFinderParams.minArea = 300
pattern_localizer.circleFinderParams.maxArea = 4000
#pattern_localizer.setPatternIDdictionary(np.load("patDictData.npy"))
pattern_localizer.setPatternData(8, 21, 0.005)
pattern_localizer.setStereoCalibration(stereoCalib)

# Load images
image_left_fn = "image_left_1.png"
image_right_fn = "image_right_1.png"
image_left = cv.imread(image_left_fn)
image_right = cv.imread(image_right_fn)

# Calculate cam->pattern pose Hc with plane fit
Hc, points2dLeft, points2dRight, points3dInLeftCamCoord = pattern_localizer.calcCamPoseViaPlaneFit(image_left, image_right, "left", False)

print("cam->pattern pose:")
print(Hc)
