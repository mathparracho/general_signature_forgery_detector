import numpy as np
from skimage import transform, img_as_ubyte
from skimage.filters import threshold_otsu

from generate_cascas import utils


def generate_cascas(img):
    img = img < threshold_otsu(img)

    coords = np.argwhere(img)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    img = img[y0:y1, x0:x1]

    img = img_as_ubyte(img.astype(float))
    h, w = img.shape

    scale_factor = min(512 / h, 512 / w)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    img_resized = transform.resize(
        img, (new_h, new_w), order=0, anti_aliasing=False, preserve_range=True
    )

    pad_vert = 512 - new_h
    pad_horiz = 512 - new_w
    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left = pad_horiz // 2
    pad_right = pad_horiz - pad_left

    img = np.pad(
        img_resized,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    img_skel, img = utils.pruning(img)
    rot = img

    sup_bin = utils.cascaS_binary(rot)
    inf_bin = utils.cascaI_binary(rot)
    res_bin = utils.res_binarizada(rot - (sup_bin | inf_bin))

    cascaS1, cascaI1 = utils.img_to_casca_func(sup_bin)
    cascaS2, cascaI2 = utils.img_to_casca_func(inf_bin)
    resS, resI = utils.img_to_casca_func(res_bin)

    TARGET_LEN = 512

    def pad(v):
        v = np.array(v)
        if len(v) > TARGET_LEN:
            return v[:TARGET_LEN]
        return np.pad(v, (0, TARGET_LEN - len(v)))

    matrix = np.vstack(
        [
            pad(cascaS1),
            pad(cascaI1),
            pad(cascaS2),
            pad(cascaI2),
            pad(resS),
            pad(resI),
        ]
    )

    return matrix.astype(np.float32), rot


def main():
    import matplotlib.pyplot as plt

    matrix, rot = generate_cascas("original_2_9.png")
    np.savetxt("casca.csv", matrix, delimiter=",", fmt="%d")
    print("Saved cascas matrix to casca.csv")

    fig, axs = plt.subplots(7, 1, figsize=(10, 12))
    axs[0].imshow(rot, cmap="gray")
    axs[0].set_title("Mask fed to casca")
    axs[0].axis("off")

    names = ["supS", "infS", "supI", "infI", "resS", "resI"]
    for ax, name, row in zip(axs[1:], names, matrix):
        ax.plot(row)
        ax.set_ylim(0, rot.shape[0])
        ax.set_title(name)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
