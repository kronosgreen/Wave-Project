import numpy as np
from __future__ import division
import sys
import os
import math
from collections import deque
from numpy import random
from scipy import misc
from matplotlib import pyplot as plt
import cv2 as cv


class Section(object):
    def __init__(self, points, birth):
        self.name = _generate_name()
        self.points = points
        self.birth = birth
        self.axis_angle = 5.0
        self.centroid = _get_centroid(self.points)
        self.centroid_vec = deque([self.centroid],
                                  maxlen=21)
        self.original_axis = _get_standard_form_line(self.centroid,
                                                     self.axis_angle)
        self.searchroi_coors = _get_searchroi_coors(self.centroid,
                                                    self.axis_angle,
                                                    15,
                                                    320)
        self.boundingbox_coors = np.int0(cv.boxPoints(
                                            cv.minAreaRect(points)))
        self.displacement = 0
        self.max_displacement = self.displacement
        self.displacement_vec = deque([self.displacement],
                                      maxlen=21)
        self.mass = len(self.points)
        self.max_mass = self.mass
        self.recognized = False
        self.death = None


    def update_searchroi_coors(self):
        self.searchroi_coors = _get_searchroi_coors(self.centroid,
                                                    self.axis_angle,
                                                    15,
                                                    320)
    def update_death(self, frame_number):
        if self.points is None:
            self.death = frame_number

    def update_points(self, frame):
        # make a polygon object of the wave's search region
        rect = self.searchroi_coors
        poly = np.array([rect], dtype=np.int32)

        # make a zero valued image on which to overlay the roi polygon
        img = np.zeros((180, 320),
                       np.uint8)

        # fill the polygon roi in the zero-value image with ones
        img = cv2.fillPoly(img, poly, 255)

        # bitwise AND with the actual image to obtain a "masked" image
        res = cv2.bitwise_and(frame, frame, mask=img)

        # all points in the roi are now expressed with ones
        points = cv2.findNonZero(res)

        # update points
        self.points = points

    def update_centroid(self):
        """Calculates the center of mass of all positive pixels that
        represent the wave, using first-order moments.
        See _get_centroid.

        Args:
          NONE

        Returns:
          NONE: updates wave.centroid
        """
        self.centroid = _get_centroid(self.points)

        # Update centroid vector.
        self.centroid_vec.append(self.centroid)


    def update_boundingbox_coors(self):
        """Finds minimum area rectangle that bounds the points of the
        wave. Returns four coordinates of the bounding box.  This is
        primarily for visualization purposes.

        Args:
          NONE

        Returns:
          NONE: updates self.boundingbox_coors attribute
        """
        boundingbox_coors = None

        if self.points is not None:
            # Obtain the moments of the object from its points array.
            X = [p[0][0] for p in self.points]
            Y = [p[0][1] for p in self.points]
            mean_x = np.mean(X)
            mean_y = np.mean(Y)
            std_x = np.std(X)
            std_y = np.std(Y)

            # We only capture points without outliers for display
            # purposes.
            points_without_outliers = np.array(
                                       [p[0] for p in self.points
                                        if np.abs(p[0][0]-mean_x) < 3*std_x
                                        and np.abs(p[0][1]-mean_y) < 3*std_y])

            rect = cv2.minAreaRect(points_without_outliers)
            box = cv2.boxPoints(rect)
            boundingbox_coors = np.int0(box)

        self.boundingbox_coors = boundingbox_coors


    def update_displacement(self):
        """Evaluates orthogonal displacement compared to original axis.
        Updates self.max_displacement if necessary.  Appends new
        displacement to deque.

        Args:
          NONE

        Returns:
          NONE: updates self.displacement and self.max_displacement
                attributes
        """
        if self.centroid is not None:
            self.displacement = _get_orthogonal_displacement(
                                                        self.centroid,
                                                        self.original_axis)

        # Update max displacement of the wave if necessary.
        if self.displacement > self.max_displacement:
            self.max_displacement = self.displacement

        # Update displacement vector.
        self.displacement_vec.append(self.displacement)


    def update_mass(self):
        """Calculates mass of the wave by weighting each pixel in a
        search roi equally and performing a simple count.  Updates
        self.max_mass attribute if necessary.

        Args:
          wave: a Section object

        Returns:
          NONE: updates self.mass and self.max_mass attributes
        """
        self.mass = _get_mass(self.points)

        # Update max_mass for the wave if necessary.
        if self.mass > self.max_mass:
            self.max_mass = self.mass


    def update_recognized(self):
        if self.recognized is False:
            if self.max_displacement >= 10 \
               and self.max_mass >= 200:
                self.recognized = True

# Get Test Video
cap = cv.VideoCapture('scene1.mp4')

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Grab First Frame w/ Grayscale Vers.
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
ret, old_thresh = cv.threshold(old_gray, 230, 255, cv.THRESH_BINARY)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
mask[..., 1] = 255

while True:
    # Get Current Frame & Grayscale Vers.
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #
    # Preprocessing
    #

    # Threshold image to only show the white part of waves
    ret, thresh = cv.threshold(gray, 215, 255, cv.THRESH_BINARY)

    # Denoise Thresholded image
    thresh = cv.fastNlMeansDenoising(thresh, 10)

    # Remove Small Artifacts
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))

    #
    # Get Sections
    #

    sections = []

    contours, hierarchy = cv.findContours(
        image=thresh,
        mode=cv.RETR_EXTERNAL,
        method=cv.CHAIN_APPROX_NONE,
        hierarchy=None,
        offset=None)

    for contour in contours:
        if keep_contour(contour):
            sections.append(Section(contour, frame_num))

    # Show Versions of Frame
    # cv.imshow("original", frame)
    cv.imshow("thresholded", thresh)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    old_frame = frame.copy()
    old_gray = gray.copy()
    old_thresh = thresh.copy()

cap.release()
cv.destroyAllWindows()