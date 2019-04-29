#
#   Wave Project
#   Description: Using OpenCV to measure ocean wave properties
#   Extended from https://github.com/citrusvanilla/multiplewavetracking_py
#   Date: 4/27/19
#   Authors: Christopher Medrano & Igor Mirsalikhov
#

import mwt_functions as mwt
import numpy as np
import cv2 as cv

# Get Test Video
cap = cv.VideoCapture('scene1.mp4')
last_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
frame_rate = int(cap.get(cv.CAP_PROP_FPS))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Grab First Frame w/ Grayscale Vers.
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
ret, old_thresh = cv.threshold(old_gray, 230, 255, cv.THRESH_BINARY)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
mask[..., 1] = 255

# Create Initial Variables
frame_num = 1
tracked_waves = []
recognized_waves = []
period = []

while True:
    # Get Current Frame & Grayscale Vers.
    ret, frame = cap.read()

    # Check if at end
    if not ret:
        break

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    #
    # Preprocessing
    #

    # Threshold image to only show the white part of waves
    ret, thresh = cv.threshold(gray, 205, 255, cv.THRESH_BINARY)

    # Resize Image
    analyze_frame = cv.resize(thresh, None,fx=0.25,fy=0.25,interpolation=cv.INTER_AREA)

    # Denoise Thresholded image
    analyze_frame = cv.fastNlMeansDenoising(analyze_frame, 5)

    # Remove Small Artifacts
    analyze_frame = cv.morphologyEx(analyze_frame, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))

    #
    # Get Waves (Detections)
    #

    sections = []

    _, contours, hierarchy = cv.findContours(
        image=analyze_frame,
        mode=cv.RETR_EXTERNAL,
        method=cv.CHAIN_APPROX_NONE,
        hierarchy=None,
        offset=None)

    for contour in contours:
        if mwt.keep_contour(contour):
            sections.append(mwt.Section(contour, frame_num))

    #
    # Update Waves (Tracking)
    #

    wave_points = []
    wave_slopes = []
    for wave in tracked_waves:
        wave.update_searchroi_coors()
        wave.update_points(analyze_frame)
        if wave.update_death(frame_rate, frame_num):
            print("Wave Traveled " + str(wave.travel_dist) +
                  "m in " + str(wave.time_alive) + "s")
        if frame_num == last_frame:
            wave.death = frame_num
        if wave.update_centroid(frame_rate):
            period.append(frame_num)
        wave.update_boundingbox_coors()
        wave.update_displacement()
        wave.update_mass()
        wave.update_recognized()
        wave_points.append(wave.centroid)
        wave_slopes.append(wave.birth_slope)

    dead_recognized_waves = [wave for wave in tracked_waves
                             if wave.death is not None
                             and wave.recognized is True]
    recognized_waves.extend(dead_recognized_waves)

    tracked_waves = [wave for wave in tracked_waves if wave.death is None]

    tracked_waves.sort(key=lambda x: x.birth, reverse=True)

    for wave in tracked_waves:
        other_waves = [wav for wav in tracked_waves if not wav == wave]
        if mwt.will_be_merged(wave, other_waves):
            wave.death = frame_num
    tracked_waves = [wave for wave in tracked_waves if wave.death is None]
    tracked_waves.sort(key=lambda x: x.birth, reverse=False)

    #
    # Calculate Stats (Recognition)
    #

    for section in sections:
        if not mwt.will_be_merged(section, tracked_waves):
            tracked_waves.append(section)

    # Draw detection boxes on original frame for visualization.
    frame = mwt.draw(tracked_waves,
                          frame
                          , 1 / 0.25)

    #
    # Show Updated Frames
    #

    # Get diff. between each wave reaching point
    if len(period) > 1:
        avg_slope = np.mean(wave_slopes)
        wavelengths = [mwt.calc_dist_lines(j, i, avg_slope) for i, j in zip(wave_points[:-1], wave_points[1:])]
        avg_wavelength = np.round(np.mean(wavelengths), 2)
        periods = [(j-i)/frame_rate for i, j in zip(period[:-1], period[1:])]
        avg_period = np.round(np.mean(periods), 2)
        temp_freq = np.round(1/avg_period, 2)
        main_stats = "Stats\nWavelength: {}m\nPeriod: {}s\nTemporal Frequency: {} wave/s"\
            .format(np.round(avg_wavelength,2), avg_period, temp_freq)
    else:
        main_stats = "Stats\nWavelength: n/a\nPeriod: n/a\nTemporal Frequency: n/a"
    for i, j in enumerate(main_stats.split('\n')):
        frame = cv.putText(
                    frame,
                    text=j,
                    org=(20, (50 + i * 45)),
                    fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1.5,
                    color=(200, 200, 200),
                    thickness=2,
                    lineType=cv.LINE_AA)

    cv.imshow("original", frame)
    cv.imshow("thresholded", analyze_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    old_frame = frame.copy()
    old_thresh = analyze_frame.copy()

    frame_num = frame_num + 1

cap.release()
cv.destroyAllWindows()