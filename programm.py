import numpy as np
import cv2 as cv
import scipy.io.wavfile as wav
import wave
from skimage.measure import compare_ssim

global Audio_shape

# region DISCRETE WAVELET TRANSFORM

def _IWT(array):
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    output[0:x, 0:ny] = (array[0::2, 0:ny] + array[1::2, 0:ny])//2
    output[x:nx, 0:ny] = array[0::2, 0:ny] - array[1::2, 0:ny]
    return output


def _IIWT(array):
    output = np.zeros_like(array)
    nx, ny = array.shape
    x = nx // 2
    output[0::2, 0:ny] = array[0:x, 0:ny] + (array[x:nx, 0:ny] + 1)//2
    output[1::2, 0:ny] = output[0::2, 0:ny] - array[x:nx, 0:ny]
    return output


def IWT2(array):
    return _IWT(_IWT(array.astype(int)).T).T


def IIWT2(array):
    return _IIWT(_IIWT(array.astype(int).T).T)


def applyImageTransform(red_plane, green_plane, blue_plane, inverse):
    if inverse:
        red_out = IIWT2(red_plane)
        green_out = IIWT2(green_plane)
        blue_out = IIWT2(blue_plane)
    else:
        red_out = IWT2(red_plane)
        green_out = IWT2(green_plane)
        blue_out = IWT2(blue_plane)
    return red_out, green_out, blue_out

# endregion


# region IMAGE AUXILIARY FUNCTIONS

def extractImagePlanes(image):
    red_plane = image[:, :, 0]
    green_plane = image[:, :, 1]
    blue_plane = image[:, :, 2]
    return red_plane, green_plane, blue_plane


def createImageFromPlanes(red_plane, green_plane, blue_plane, dimensions):
    output_image = np.zeros(dimensions, dtype='uint8')
    output_image[:, :, 0] = red_plane
    output_image[:, :, 1] = green_plane
    output_image[:, :, 2] = blue_plane
    return output_image

# endregion


# region AUDIO AUXILIARY FUNCTIONS

def loadAudio(filename):
    sample, audio_data = wav.read(filename)
    x, y = audio_data.shape
    if x % 2 == 1:
        audio_data = np.delete(audio_data, x - 1, axis=0)
    return sample, audio_data


def saveAudio(filename, sample, data):
    with wave.open(filename, 'wb') as out_wav:
        out_wav.setnchannels(2)
        out_wav.setsampwidth(1)
        out_wav.setframerate(sample)
        out_wav.writeframesraw(data)

# endregion


# region STEGO FUNCTIONS

def encode(image_name, audio_name):
    input_image = cv.imread(image_name)
    cv.imshow('INPUT IMAGE', input_image)
    red_plane, green_plane, blue_plane = extractImagePlanes(input_image)
    image_length, image_height = red_plane.shape
    audio_length = image_length // 2
    audio_height = image_height // 4
    red_plane, green_plane, blue_plane = applyImageTransform(red_plane, green_plane, blue_plane, False)

    fs, audio = loadAudio(audio_name)
    audio_dim, audio_channel = audio.shape
    
    out_audio_data = np.resize(audio, (2*audio_length, audio_height, 3))
    audio_r, audio_g, audio_b = extractImagePlanes(out_audio_data)

    image_after_transform = createImageFromPlanes(red_plane, green_plane, blue_plane, (image_length, image_height, 3))
    cv.imshow("IMAGE AFTER TRANSFORMATION", image_after_transform)
    cv.imwrite("after_transform.jpg", image_after_transform)

    for i in range(audio_height):
        for j in range(audio_length):
            bitRH = audio_r[j][i] >> 4
            bitRL = audio_r[j][i] & 15
            red_plane[i][j] = (red_plane[i][j] & 240) | bitRH
            red_plane[i+audio_height][j] = (red_plane[i+audio_height][j] & 240) | bitRL

            bitGH = audio_g[j][i] >> 4
            bitGL = audio_g[j][i] & 15
            green_plane[i][j] = (green_plane[i][j] & 240) | bitGH
            green_plane[i+audio_height][j] = (green_plane[i+audio_height][j] & 240) | bitGL

            bitBH = audio_b[j][i] >> 4
            bitBL = audio_b[j][i] & 15
            blue_plane[i][j] = (blue_plane[i][j] & 240) | bitBH
            blue_plane[i+audio_height][j] = (blue_plane[i+audio_height][j] & 240) | bitBL

    red_plane, green_plane, blue_plane = applyImageTransform(red_plane, green_plane, blue_plane, True)
    stego_image = createImageFromPlanes(red_plane, green_plane, blue_plane, (image_length, image_height, 3))
    cv.imshow("STEGO IMAGE", stego_image)
    cv.imwrite("stego.jpg", stego_image)
    measurement_image(input_image, stego_image)
    return stego_image


def decode(image):
    red_plane, green_plane, blue_plane = extractImagePlanes(image)
    red_plane, green_plane, blue_plane = applyImageTransform(red_plane, green_plane, blue_plane, False)

    image_length, image_height = red_plane.shape
    audio_length = image_length // 2
    audio_height = image_height // 4

    audio = np.zeros((audio_length, audio_height, 3))
    audio_r, audio_g, audio_b = extractImagePlanes(audio)

    for i in range(audio_height):
        for j in range(audio_length):
            bitRH = (red_plane[i][j] & 15) << 4
            bitRL = red_plane[i+audio_height][j] & 15
            audio_r[j][i] = bitRH | bitRL

            bitGH = (green_plane[i][j] & 15) << 4
            bitGL = green_plane[i+audio_height][j] & 15
            audio_g[j][i] = bitGH | bitGL

            bitBH = (blue_plane[i][j] & 15) << 4
            bitBL = blue_plane[i+audio_height][j] & 15
            audio_b[j][i] = bitBH | bitBL

    audio = createImageFromPlanes(audio_r, audio_g, audio_b, (audio_length, audio_height, 3))
    audio = np.resize(audio, Audio_shape)

    return audio

# endregion


# region MEASUREMENT FUNCTIONS

def psnr(img1, img2):
    diff = img1 - img2
    mse = np.mean(np.square(diff))
    psnr_value = 10 * np.log10(255 * 255 / mse)
    print("PSRN:")
    print(psnr_value)
    return 0


def ssim(img1, img2):
    (grayScore, diff) = compare_ssim(img1, img2, full=True, multichannel=True)
    print("SSIM:")
    print(grayScore*100)
    return 0


def measurement_image(img1, img2):
    psnr(img1, img2)
    ssim(img1, img2)
    return 0

#endregion


# region MAIN

fs, audio = loadAudio('godfather_final.wav')
Audio_shape = audio.shape

stego = encode('input_image_1_1000x1000.jpg', 'godfather_final.wav')
out_data = decode(stego)

saveAudio('proba.wav', fs, out_data)

# endregion

cv.waitKey(0)
cv.destroyAllWindows()