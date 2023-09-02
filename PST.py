import math
import numpy as np
import mahotas as mh


# Define functions
def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def PST(I, LPF= 0.21, Phase_strength=0.48 , Warp_strength=12.14, Threshold_min= -1, Threshold_max= 0.0019, Morph_flag =1):
    L = 0.5
    x = np.linspace(-L, L, I.shape[0])
    y = np.linspace(-L, L, I.shape[1])
    [X1, Y1] = (np.meshgrid(x, y))
    X = X1.T
    Y = Y1.T
    [THETA, RHO] = cart2pol(X, Y)

    # Apply localization kernel to the original image to reduce noise
    Image_orig_f = ((np.fft.fft2(I)))
    expo = np.fft.fftshift(np.exp(-np.power((np.divide(RHO, math.sqrt((LPF ** 2) / np.log(2)))), 2)))
    Image_orig_filtered = np.real(np.fft.ifft2((np.multiply(Image_orig_f, expo))))
    # Constructing the PST Kernel
    PST_Kernel_1 = np.multiply(np.dot(RHO, Warp_strength), np.arctan(np.dot(RHO, Warp_strength))) - 0.5 * np.log(
        1 + np.power(np.dot(RHO, Warp_strength), 2))
    PST_Kernel = PST_Kernel_1 / np.max(PST_Kernel_1) * Phase_strength
    # Apply the PST Kernel
    temp = np.multiply(np.fft.fftshift(np.exp(-1j * PST_Kernel)), np.fft.fft2(Image_orig_filtered))
    Image_orig_filtered_PST = np.fft.ifft2(temp)

    # Calculate phase of the transformed image
    PHI_features = np.angle(Image_orig_filtered_PST)

    if Morph_flag == 0:
        out = PHI_features
    else:
        #   find image sharp transitions by thresholding the phase
        features = np.zeros((PHI_features.shape[0], PHI_features.shape[1]))
        features[PHI_features > Threshold_max] = 1  # Bi-threshold decision
        features[PHI_features < Threshold_min] = 1  # as the output phase has both positive and negative values
        features[I < (np.amax(I) / 20)] = 0  # Removing edges in the very dark areas of the image (noise)

        # apply binary morphological operations to clean the transformed image
        out = features
        out = mh.thin(out, 1)
        out = mh.bwperim(out, 4)
        out = mh.thin(out, 1)
        out = mh.erode(out, np.ones((1, 1)));

    return out