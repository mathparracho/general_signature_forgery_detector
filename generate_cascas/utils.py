"""Utility functions to work with signature shells (cascas)."""

import numpy as np
from skimage import morphology, measure, transform


def pruning(bw: np.ndarray):
    bw = morphology.binary_opening(bw)
    bw = morphology.remove_small_holes(bw)
    return morphology.skeletonize(bw), bw.astype(np.uint8)



def casca(bw: np.ndarray):
    """Return matrices of the top-shell and bottom-shell."""
    superior = np.zeros_like(bw, dtype=np.uint8)
    inferior = np.zeros_like(bw, dtype=np.uint8)
    for j in range(bw.shape[1]):
        rows = np.where(bw[:, j])[0]
        if rows.size:
            superior[rows[0], j] = 1
            inferior[rows[-1], j] = 1
    return superior, inferior


def img_to_casca_func(img: np.ndarray):
    """Return superior and inferior shell functions from a binary image."""
    img = np.flipud(img)
    casca_s = np.zeros(img.shape[1], dtype=int)
    casca_i = np.zeros_like(casca_s)
    for j in range(img.shape[1]):
        rows = np.where(img[:, j])[0]
        if rows.size:
            casca_i[j] = rows[0]
            casca_s[j] = rows[-1]
    return casca_s, casca_i


def casca_binary(img: np.ndarray) -> np.ndarray:
    """Mimic ``f_casca_Bin`` from the MATLAB code."""
    result = np.zeros_like(img, dtype=np.uint8)
    for j in range(img.shape[1]):
        rows = np.where(img[:, j])[0]
        if rows.size:
            first = rows[0]
            result[first:, j] = img[first:, j]
    return result


def cascaS_binary(img: np.ndarray) -> np.ndarray:
    """Approximation of ``f_cascaS_Bin``."""
    result = np.zeros_like(img, dtype=np.uint8)
    for j in range(img.shape[1]):
        rows = np.where(img[:, j])[0]
        if rows.size:
            idx = rows[0]
            while idx < img.shape[0] and img[idx, j]:
                result[idx, j] = 1
                idx += 1
    return result


def cascaI_binary(img: np.ndarray) -> np.ndarray:
    """Approximation of ``f_cascaI_Bin``."""
    result = np.zeros_like(img, dtype=np.uint8)
    for j in range(img.shape[1] - 1, -1, -1):
        rows = np.where(img[:, j])[0]
        if rows.size:
            idx = rows[-1]
            while idx >= 0 and img[idx, j]:
                result[idx, j] = 1
                idx -= 1
    return result


def sup_binarizada(img: np.ndarray):
    """Equivalent to ``f_Sup_Binarizada``."""
    img = np.flipud(img)
    thickness = np.zeros(img.shape[1], dtype=int)
    casca_bin = np.zeros_like(img, dtype=np.uint8)
    for j in range(img.shape[1] - 1, -1, -1):
        contador = 0
        for i in range(img.shape[0] - 1, -1, -1):
            if img[i, j]:
                idx = i
                while idx >= 0 and img[idx, j]:
                    casca_bin[j, idx] = idx
                    contador += 1
                    idx -= 1
                break
        thickness[j] = contador
    return casca_bin, thickness


def inf_binarizada(img: np.ndarray):
    """Equivalent to ``f_Inf_Binarizada``."""
    img = np.flipud(img)
    thickness = np.zeros(img.shape[1], dtype=int)
    casca_bin = np.zeros_like(img, dtype=np.uint8)
    for j in range(img.shape[1]):
        contador = 0
        for i in range(img.shape[0]):
            if img[i, j]:
                idx = i
                while idx < img.shape[0] and img[idx, j]:
                    casca_bin[j, idx] = idx
                    contador += 1
                    idx += 1
                break
        thickness[j] = contador
    return casca_bin, thickness


def res_binarizada(img: np.ndarray) -> np.ndarray:
    """Equivalent to ``f_Res_Binarizada``."""
    img = np.fliplr(img)
    casca_res = np.zeros_like(img, dtype=np.uint8)
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            idx = i
            while idx < img.shape[0] and img[idx, j]:
                casca_res[idx, j] = 1
                idx += 1
    return casca_res


def calculate_centroids(image: np.ndarray):
    labeled = measure.label(image)
    props = measure.regionprops(labeled)
    if not props:
        raise ValueError("No regions found")
    prop = max(props, key=lambda p: p.area)
    y1, x1 = prop.centroid
    x1_int = int(round(x1))
    left_half = image[:, :x1_int]
    right_half = image[:, x1_int:]
    props_left = measure.regionprops(measure.label(left_half))
    props_right = measure.regionprops(measure.label(right_half))
    if not props_left or not props_right:
        raise ValueError("Could not compute centroids")
    yL, xL = props_left[0].centroid
    yR, xR = props_right[0].centroid
    return x1, y1, xL, yL, xR, yR


def rotate_image(image: np.ndarray, x: float, xR: float, xL: float, yL: float, yR: float):
    point1 = np.array([xL, yL])
    point2 = np.array([xR + x, yR])
    slope = (yR - yL) / ((xR + x) - xL)
    angle = 0.5 * np.degrees(np.arctan(slope))
    rot = transform.rotate(image, angle, resize=True, order=3)
    props = measure.regionprops(measure.label(rot.astype(bool)))
    if props:
        prop = props[0]
        minr, minc, maxr, maxc = prop.bbox
        rot = rot[minr:maxr, minc:maxc]
    return rot, angle
