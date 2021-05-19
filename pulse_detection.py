# FILE NAME: pulse_detection.py
#
# DESCRIPTION: file for remote pulse detection
#
# REVISION DATE: 04/23/2021
#
# input: config.txt
#
# output:
#   - forehead_fft.png: power spectral density plot (PSD) of forehead
#   - left_background_fft.png:
#   - right_background_fft.png
#   - mean_background_fft.png
#   - forehead_background_combined_fft.png
#   - heat_map.jpg
#
# Carnegie Mellon Cylab

#import libraries
import numpy as np
import sklearn.decomposition
import cv2
import os
import warnings
import sys
import matplotlib.pyplot as plt
from scipy.fft import fft as ft
from scipy.fft import fftfreq

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.utils.testing import ignore_warnings

FPS = 0
MIN_HR_BPM = 40.0
MAX_HR_BMP = 240.0
MIN_HR_FREQ = MIN_HR_BPM/60.0
MAX_HR_FREQ = MAX_HR_BMP/60.0

#load haar_model
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

cascPath = haar_model
faceCascade = cv2.CascadeClassifier(cascPath)


def distance(roi1, roi2):
    return sum((roi1[i] - roi2[i]) ** 2 for i in range(len(roi1)))

#input frame,
def get_rois(frame, previous_face, i, heat_map):
    forehead_roi = []
    background_left = []
    background_right = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    roi = None
    curr_face = None

    # If no face detected, use ROI from previous frame
    if len(faces) == 0:
        curr_face = previous_face
    elif len(faces) == 1:
        curr_face = faces[0]
    else:
        if previous_face is not None:
            closest_distance = float("inf")
            for face in faces:
                if distance(previous_face, face) < closest_distance:
                    curr_face = face
        else:
            largest_size = 0
            for face in faces:
                if (face[2] * face[3]) > largest_size:
                    # largest_size = face[2] * face[3]
                    curr_face = face

    if curr_face is not None:
        (x, y, w, h) = curr_face
        height, width, _ = frame.shape

        #establish boundaries for background frames
        br_boundaries = (min(y+100, height), min(x+w+50, width-105))
        bl_boundaries = (min(y+100, height), max(5, x-150))
        background_left = frame[y:bl_boundaries[0], bl_boundaries[1]:x-50]
        background_right = frame[y:br_boundaries[0], br_boundaries[1]:br_boundaries[1]+100]

        #add middle of forehead to heat map image
        heat_map[50:h + 50, i:i + 1] = frame[y:y + h, x + (w // 2):x + (w // 2) + 1]

        #find forehead
        zoomed_face_offset = int(0.3 * w)
        zoomed_face = (x + zoomed_face_offset, y, int(0.4 * w), int(0.25 * h))

        (x, y, w, h) = zoomed_face
        forehead_roi = frame[y:y + h, x:x + w]

        # create zoomed face mask
        mask = np.full(frame.shape, True, dtype=bool)
        mask[y:y + h, x:x + w, :] = False
        roi = np.ma.array(frame, mask=mask)

    return curr_face, roi, heat_map, forehead_roi, background_left, background_right

@ignore_warnings(category=ConvergenceWarning)
def get_power_spec_pulse(rgb_channels, vidcap):
    FPS = vidcap.get(cv2.CAP_PROP_FPS)
    NUM_FRAMES = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    ica = sklearn.decomposition.FastICA(n_components= 1)
    ica_sig = ica.fit_transform(rgb_channels)
    ica_sig = ica_sig[:,0]

    # normalize
    ica_mean = np.mean(ica_sig)
    ica_std = np.std(ica_sig)
    normalized_ica = (ica_sig - ica_mean) / ica_std

    #scipy fourier transform
    yf = ft(normalized_ica)
    xf = fftfreq(NUM_FRAMES, 1 / FPS)

    validIdx = np.where((xf > MIN_HR_FREQ) & (xf <= MAX_HR_FREQ))
    validyf = yf[validIdx]
    validxf = xf[validIdx]

    validpwr = np.abs(validyf) ** 2

    return validxf, validpwr


def getHeartRate(forehead_channel, bl, br, vidcap, output_path):
    # Find power spectrum
    fft, powerSpec = get_power_spec_pulse(forehead_channel, vidcap)
    fft_bl, powerSpec_bl = get_power_spec_pulse(bl, vidcap)
    fft_br, powerSpec_br = get_power_spec_pulse(br, vidcap)
    added_background = np.add(powerSpec_bl, powerSpec_br)
    mean_background = added_background / 2.0
    subtracted_power_spec = powerSpec - mean_background

    max_pspec_diff = np.max(subtracted_power_spec) - np.mean(subtracted_power_spec)

    for i in range(6):
        fft_tmp, powerSpec_tmp = get_power_spec_pulse(forehead_channel, vidcap)
        fft_bl_tmp, powerSpec_bl_tmp = get_power_spec_pulse(bl, vidcap)
        fft_br_tmp, powerSpec_br_tmp = get_power_spec_pulse(br, vidcap)
        added_background_tmp = np.add(powerSpec_bl_tmp, powerSpec_br_tmp)
        mean_background_tmp = added_background_tmp/2.0
        subtracted_power_spec_tmp = powerSpec_tmp - mean_background_tmp
        pspec_diff = np.max(subtracted_power_spec_tmp) - np.mean(subtracted_power_spec_tmp)

        if (pspec_diff > max_pspec_diff):
            max_pspec_diff = pspec_diff
            fft, powerSpec = fft_tmp, powerSpec_tmp
            fft_bl, powerSpec_bl = fft_bl_tmp, powerSpec_bl_tmp
            fft_br, powerSpec_br = fft_br_tmp, powerSpec_br_tmp
            added_background = added_background_tmp
            mean_background = mean_background_tmp
            subtracted_power_spec = subtracted_power_spec_tmp



    #make plots
    plt.plot(fft_bl, powerSpec_bl)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title('Left Background Fourier Transform')
    plt.savefig(output_path + '/left_background_fft.png')
    plt.close()

    plt.plot(fft_br, powerSpec_br)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title('Right Background Fourier Transform')
    plt.savefig(output_path + '/right_background_fft.png')
    plt.close()

    plt.plot(fft_bl, mean_background)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title('Mean Background Fourier Transform')
    plt.savefig(output_path + '/mean_background_fft.png')
    plt.close()

    plt.plot(fft, powerSpec)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title('Forehead Fourier Transform')
    plt.savefig(output_path + '/forehead_fft.png')
    plt.close()

    hr_freq = fft[np.argmax(subtracted_power_spec)]

    plt.plot(fft_bl, subtracted_power_spec)
    plt.axvline(x=hr_freq, linewidth = 0.5)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    rounded_hr = round(hr_freq * 60.0, 3)
    plt.text(MAX_HR_FREQ, max(subtracted_power_spec), 'Pulse: ' + str(rounded_hr) + ' BPM',
            horizontalalignment='right',
            verticalalignment='top')
    plt.title('Forehead/Background Combined Fourier Transform')
    plt.savefig(output_path + '/forehead_background_combined_fft.png')
    plt.close()


    hr = hr_freq * 60.0
    return hr


def heart_rate_extraction(filepath, output_path):
    vidcap = cv2.VideoCapture(filepath)
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    #Your output video with face tracking block
    heat_map = np.zeros((height, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), 3))

    colorSig = [] # Will store the average RGB color values in each frame's ROI
    bl_signal = []
    br_signal = []

    previousFaceBox = None

    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = vidcap.read()
        if not ret:
            break

        previousFaceBox, roi, heat_map, forehead_roi, background_left, background_right = get_rois(frame, previousFaceBox, i, heat_map)

        if (roi is not None) and (np.size(roi) > 0):
            colorChannels = forehead_roi.reshape(-1, forehead_roi.shape[-1])
            bl_channels = background_left.reshape(-1, background_left.shape[-1])
            br_channels = background_right.reshape(-1, background_right.shape[-1])


            colorSig.append(np.mean(colorChannels, axis= 0))
            bl_signal.append(np.mean(bl_channels, axis= 0))
            br_signal.append(np.mean(br_channels, axis= 0))

        i += 1
        cv2.waitKey(1)

    cv2.imwrite(output_path + "/heat_map.jpg", heat_map[50:150, :-20].astype(np.uint8))
    calculated_hr = getHeartRate(np.array(colorSig), np.array(bl_signal), np.array(br_signal), vidcap, output_path)

    return calculated_hr


if __name__=='__main__':
    config_file = sys.argv[-1]
    config = open(config_file, "r")
    config_contents = config.read().split("\n")

    filepath = os.getcwd() + config_contents[0].split(' ')[-1]
    output_path = os.getcwd() + config_contents[1].split(' ')[-1]
    MIN_HR_BPM = float(config_contents[2].split(' ')[-1])
    MAX_HR_BMP = float(config_contents[3].split(' ')[-1])

    MIN_HR_FREQ = MIN_HR_BPM/60.0
    MAX_HR_FREQ = MAX_HR_BMP/60.0

    pulse_rate = heart_rate_extraction(filepath, output_path)
    print(pulse_rate)

    config.close()
