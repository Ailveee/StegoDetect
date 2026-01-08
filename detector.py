import cv2
import numpy as np
from PIL import Image
from scipy.stats import chisquare

def lsb_ratio(image_path):
    img = Image.open(image_path)
    pixels = np.array(img)
    lsb = pixels & 1
    return np.mean(lsb)

def chi_square_test(image_path):
    img = Image.open(image_path).convert("L")
    pixels = np.array(img).flatten()

    hist, _ = np.histogram(pixels, bins=256, range=(0,256))

    even = hist[::2].astype(float)
    odd = hist[1::2].astype(float)

    # Safety checks
    if even.sum() == 0 or odd.sum() == 0:
        return 1.0  # not suspicious

    even = even * (odd.sum() / even.sum())

    _, p = chisquare(f_obs=odd, f_exp=even)

    if np.isnan(p) or np.isinf(p):
        return 1.0

    return p


def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    hist = cv2.calcHist([img], [0], None, [64], [0,256])
    hist = hist.flatten()
    hist /= hist.sum()

    mean = np.mean(img)
    std = np.std(img)
    lsb = lsb_ratio(image_path)

    return np.concatenate(([mean, std, lsb], hist))

def detect(image_path, model):
    features = extract_features(image_path)
    ml_prob = model.predict_proba([features])[0][1]

    chi_p = chi_square_test(image_path)
    lsb = lsb_ratio(image_path)

    score = (0.6 * ml_prob) + (0.3 * (1 - chi_p)) + (0.1 * abs(lsb - 0.5))

    # SAFETY: handle NaN and clamp
    if np.isnan(score) or np.isinf(score):
        score = ml_prob

    score = max(0.0, min(1.0, score))

    verdict = "STEGO" if score > 0.5 else "CLEAN"
    return verdict, round(score, 2)

