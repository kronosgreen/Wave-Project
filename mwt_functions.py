from __future__ import division
import numpy as np
import sys
import os
import math
from collections import deque
from numpy import random
from scipy import misc
from scipy.stats import linregress
from matplotlib import pyplot as plt
import cv2 as cv

#
#   Functions for Wave Class
#   Extended from : https://github.com/citrusvanilla/multiplewavetracking_py
#   Christopher Medrano & Igor Mirsalikhov
#   Date: 4/29/19
#


# Used to get Wave Length
# P1 is birth centroid
# P2 is death centroid
# Slope refers to slope of wave's trend line at birth
def calc_dist_lines(p1, p2, slope):
    if None in (p1, p2, slope):
        return 0
    # Perpendicular slope
    perp_slope = -1 / slope
    # B for Perpendicular line
    pB = p1[1] - p1[0] * perp_slope
    # B for line 2
    line_2_B = p2[1] - p2[0] * slope
    # Intersecting point w/ Perp & Line 2
    inter_x = (line_2_B - pB) / (perp_slope - slope)
    inter_y = perp_slope * inter_x + pB
    # Calculate Distance
    return np.sqrt((p1[1] - inter_y) ** 2 + (p1[0] - inter_x) ** 2) * 1.25


# Check if wave has passed point
# Used to calculate Period
def test_passed_point(m, b, test_p):
    # See if test point is above current wave line
    if test_p[1] < m * test_p[0] + b:
        return True
    else:
        return False


def draw_line(frame, m, b, resize_factor):
    if not (isinstance(m, float) or isinstance(b, float)):
        return frame
    p1 = (0, int(resize_factor*np.round(m*0 + b)))
    p2 = (int(np.round(resize_factor*300)), int(np.round(resize_factor*(m*300 + b))) )

    frame = cv.line(frame, p1, p2, color=(0, 255, 0), thickness=5)

    return frame


class Section(object):
    def __init__(self, points, birth):
        self.points = points
        self.birth = birth
        self.slope = np.nan
        self.intercept = 0
        self.moving_speed = 0
        self.time_alive = 0
        self.travel_dist = 0
        self.passed_point = False
        self.axis_angle = 5.0
        self.centroid = _get_centroid(self.points)
        self.centroid_at_birth = self.centroid
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

        # Calculate Slope/Intercept
        self.calc_slope(self.points)

    # Get slope of wave line
    def calc_slope(self, points):
        #X = resize_factor * np.array(points[:, 0][0])
        #Y = resize_factor * np.array(points[:, 0][1])

        #slope, intercept, r_value, p_value, std_err = linregress(X, Y)
        p1 = points[0,0]
        p2 = points[len(points)-1,0]

        if np.isnan(self.slope) or -0.1 < self.slope < 0.1:
            self.slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.intercept = p2[1] - p2[0] * self.slope

    def update_searchroi_coors(self):
        self.searchroi_coors = _get_searchroi_coors(self.centroid,
                                                    self.axis_angle,                                           15,
                                                    320)

    def update_death(self, fps, frame_number):
        self.time_alive = np.round((frame_number - self.birth) / fps, 2)
        if self.points is None:
            self.death = frame_number
            self.travel_dist = np.round(calc_dist_lines(self.centroid_at_birth, self.centroid, self.slope),2)
            return True
        else:
            return False

    def update_points(self, frame, resize_factor):
        # make a polygon object of the wave's search region
        rect = self.searchroi_coors
        poly = np.array([rect], dtype=np.int32)

        # make a zero valued image on which to overlay the roi polygon
        img = np.zeros((180, 320),
                       np.uint8)

        # fill the polygon roi in the zero-value image with ones
        img = cv.fillPoly(img, poly, 255)

        # bitwise AND with the actual image to obtain a "masked" image
        res = cv.bitwise_and(frame, frame, mask=img)

        # all points in the roi are now expressed with ones
        points = cv.findNonZero(res)

        # update points
        self.points = points

        if points is not None:
            self.calc_slope(self.points)

        return frame

    def update_centroid(self, fps):
        """Calculates the center of mass of all positive pixels that
        represent the wave, using first-order moments.
        See _get_centroid.

        Args:
          NONE

        Returns:
          NONE: updates wave.centroid
        """
        old_centroid = self.centroid
        self.centroid = _get_centroid(self.points)

        # Update centroid vector.
        self.centroid_vec.append(self.centroid)

        # Update Current Speed
        speed = np.round(calc_dist_lines(old_centroid, self.centroid, self.slope) * fps, 2)
        # Remove erroneous calculations
        if 1 < speed < 10:
            self.moving_speed = speed

        if not self.passed_point and test_passed_point(self.slope, self.intercept, (260, 150)):
            self.passed_point = True
            return True
        else:
            return False

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

            rect = cv.minAreaRect(points_without_outliers)
            box = cv.boxPoints(rect)
            boundingbox_coors = np.int0(box)

        self.boundingbox_coors = boundingbox_coors

    def update_displacement(self):
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
        self.mass = _get_mass(self.points)

        # Update max_mass for the wave if necessary.
        if self.mass > self.max_mass:
            self.max_mass = self.mass

    def update_recognized(self):
        if self.recognized is False:
            if self.max_displacement >= 10 \
               and self.max_mass >= 200:
                self.recognized = True


def keep_contour(contour,
                 area= True,
                 inertia= True,
                 min_area= 100,
                 max_area= 1e7,
                 min_inertia_ratio= 0.0,
                 max_inertia_ratio= 0.1):
    """Contour filtering function utilizing OpenCV.  In our case,
    we are looking for oblong shapes that exceed a user-defined area.

    Args:
      contour: A contour from an array of contours
      area: boolean flag to filter contour by area
      inertia: boolean flag to filter contour by inertia
      min_area: minimum area threshold for contour
      max_area: maximum area threshold for contour
      min_inertia_ratio: minimum inertia threshold for contour
      max_inertia_ratio: maximum inertia threshold for contour

    Returns:
      ret: A boolean TRUE if contour meets conditions, else FALSE
    """
    # Initialize the return value.
    ret = True

    # Obtain contour moments.
    moments = cv.moments(contour)

    # Filter Contours By Area.
    if area is True and ret is True:
        area = cv.contourArea(contour)
        if area < min_area or area >= max_area:
            ret = False

    # Filter contours by inertia.
    if inertia is True and ret is True:
        denominator = math.sqrt((2*moments['m11'])**2
                                + (moments['m20']-moments['m02'])**2)
        epsilon = 0.01
        ratio = 0.0

        if denominator > epsilon:
            cosmin = (moments['m20']-moments['m02']) / denominator
            sinmin = 2*moments['m11'] / denominator
            cosmax = -cosmin
            sinmax = -sinmin

            imin = (0.5*(moments['m20']+moments['m02'])
                    - 0.5*(moments['m20']-moments['m02'])*cosmin
                    - moments['m11']*sinmin)
            imax = (0.5*(moments['m20']+moments['m02'])
                    - 0.5*(moments['m20']-moments['m02'])*cosmax
                    - moments['m11']*sinmax)
            ratio = imin / imax
        else:
            ratio = 1

        if ratio < min_inertia_ratio or ratio >= max_inertia_ratio:
            ret = False
            #center.confidence = ratio * ratio;

    return ret


def _get_mass(points):
    """Simple function to calculate mass of an array of points with
    equal weighting of the points.

    Args:
      points: an array of non-zero points

    Returns:
      mass:  "mass" of the points
    """
    mass = 0

    if points is not None:
        mass = len(points)

    return mass


def _get_centroid(points):
    """Helper function for getting the x,y coordinates of the center of
    mass of an object that is represented by positive pixels in a
    bilevel image.

    Args:
      points: array of points
    Returns:
      centroid: 2 element array as [x,y] if points is not empty
    """
    centroid = None

    if points is not None:
        centroid = [int(sum([p[0][0] for p in points]) / len(points)),
                    int(sum([p[0][1] for p in points]) / len(points))]

    return centroid


def _get_orthogonal_displacement(point, standard_form_line):
    """Helper function to calculate the orthogonal distance of a point
    to a line.

    Args:
      point: 2-element array representing a point as [x,y]
      standard_form_line: 3-element array representing a line in
                          standard form coordinates as [A,B,C]
    Returns:
      ortho_disp: distance of point to line in pixels
    """
    ortho_disp = 0

    # Retrieve standard form coefficients of original axis.
    a = standard_form_line[0]
    b = standard_form_line[1]
    c = standard_form_line[2]

    # Retrieve current location of the wave.
    x0 = point[0]
    y0 = point[1]

    # Calculate orthogonal distance from current postion to
    # original axis.
    ortho_disp = np.abs(a*x0 + b*y0 + c) / math.sqrt(a**2 + b**2)

    return int(ortho_disp)


def _get_standard_form_line(point, angle):
    """Helper function returning a 3-element array corresponding to
    coefficients of the standard form for a line of Ax+By=C.
    Requires one point in [x,y], and a counterclockwise angle from the
    horizion in degrees.

    Args:
      point: a two-element array in [x,y] representing a point
      angle: a float representing counterclockwise angle from horizon
             of a line

    Returns:
      coefficients: a three-element array as [A,B,C]
    """
    coefficients = [None, None, None]

    coefficients[0] = np.tan(np.deg2rad(-angle))
    coefficients[1] = -1
    coefficients[2] = (point[1] - np.tan(np.deg2rad(-angle))*point[0])

    return coefficients


def _get_searchroi_coors(centroid, angle, searchroi_buffer, frame_width):
    """Helper function for returning the four coordinates of a
    polygonal search region- a region in which we would want to merge
    several independent wave objects into one wave object because they
    are indeed one wave.  Creates a buffer based on searchroi_buffer
    and the polygon (wave) axis angle.

    Args:
      centroid: a two-element array representing center of mass of
                a wave
      angle: counterclosewise angle from horizon of a wave's axis
      searchroi_buffer: a buffer, in pixels, in which to generate
                        a search region buffer
      frame_width: the width of the frame, to establish left and
                   right bounds of a polygon

    Returns:
      polygon_coors: a four element array representing the top left,
                     top right, bottom right, and bottom left
                     coordinates of a search region polygon

    """
    polygon_coors = [[None, None],
                     [None, None],
                     [None, None],
                     [None, None]]

    delta_y_left = np.round(centroid[0] * np.tan(np.deg2rad(angle)))
    delta_y_right = np.round((frame_width - centroid[0])
                             * np.tan(np.deg2rad(angle)))

    upper_left_y = int(centroid[1] + delta_y_left - searchroi_buffer)
    upper_left_x = 0
    upper_right_y = int(centroid[1] - delta_y_right - searchroi_buffer)
    upper_right_x = frame_width

    lower_left_y = int(centroid[1] + delta_y_left + searchroi_buffer)
    lower_left_x = 0
    lower_right_y = int(centroid[1] - delta_y_right + searchroi_buffer)
    lower_right_x = frame_width

    polygon_coors = [[upper_left_x, upper_left_y],
                     [upper_right_x, upper_right_y],
                     [lower_right_x, lower_right_y],
                     [lower_left_x, lower_left_y]]

    return polygon_coors


def will_be_merged(section, list_of_waves):
    """Boolean evaluating whether or not a section is in an existing
    wave's search region.

    Args:
      section: a wave object
      list_of_waves: a list of waves having search regions in which a
                     wave might fall

    Returns:
      going_to_be_merged: evaluates to True if the section is in an
                          existing wave's search region.
    """
    # All sections are initially new waves & will not be merged.
    going_to_be_merged = False

    # Find the section's major axis' projection on the y axis.
    delta_y_left = np.round(section.centroid[0]
                            * np.tan(np.deg2rad(section.axis_angle)))
    left_y = int(section.centroid[1] + delta_y_left)

    # For each existing wave, see if the section's axis falls in
    # another wave's search region.
    for wave in list_of_waves:
        if left_y >= wave.searchroi_coors[0][1] and left_y <= wave.searchroi_coors[3][1]:
            going_to_be_merged = True
            break

    return going_to_be_merged


def draw(waves, frame, resize_factor):
    """Simple function to draw on a frame for output.  Draws bounding
    boxes in accordance with wave.boundingbox_coors attribute, and draws
    some wave stats to accompany each potential wave, including whether
    or not the object is actually a wave (i.e. wave.recognized == True).

    Args:
      waves: list of waves
      frame: frame on which to draw waves
      resize_factor: factor to resize boundingbox coors to match output
                     frame.

    Returns:
      frame: input frame with waves drawn on top
    """
    # Iterate through a list of waves.

    for wave in waves:

        # For drawing circles on detected features
        # center = (wave.centroid[0],wave.centroid[1])
        # radius = 15
        # cv2.circle(frame,center,radius,(0,255,0),2)

        if wave.death is None:
            # If wave is a wave, draw green, else yellow.
            # Set wave text accordingly.
            '''if wave.recognized is True:
                drawing_color = (0, 255, 0)
                text = ("Wave Detected!\nmass: {}\ndisplacement: {}"
                        .format(wave.mass, wave.displacement))
            else:
                drawing_color = (0, 255, 255)
                text = ("Potential Wave\nmass: {}\ndisplacement: {}"
                        .format(wave.mass, wave.displacement))
            '''
            text = ("Wave\nSpeed: {}mps\nTime Alive: {}s"
                    .format(wave.moving_speed, wave.time_alive))

            if len(wave.centroid_vec) > 20:
                # Draw Bounding Boxes:
                # Get boundingbox coors from wave objects and resize.

                rect = wave.boundingbox_coors
                rect[:] = [resize_factor * rect[i] for i in range(4)]
                frame = cv.drawContours(frame, [rect], 0, (0, 255, 0), 2)

                # Use moving averages of wave centroid for stat locations
                moving_x = np.mean([wave.centroid_vec[-k][0]
                                    for k
                                    in range(1, min(20, 1 + len(wave.centroid_vec)))])
                moving_y = np.mean([wave.centroid_vec[-k][1]
                                    for k
                                    in range(1, min(20, 1 + len(wave.centroid_vec)))])
                frame = cv.circle(frame,(int(resize_factor*moving_x),int(resize_factor*moving_y)),10,(0,255,0),2)

                for i, j in enumerate(text.split('\n')):
                    frame = cv.putText(
                        frame,
                        text=j,
                        org=(int(resize_factor * moving_x)-300,
                             int(resize_factor * moving_y)-170
                             + (50 + i * 45)),
                        fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=1.5,
                        color=(220,220,220),
                        thickness=2,
                        lineType=cv.LINE_AA)
    return frame