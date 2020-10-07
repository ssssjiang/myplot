"""
2D visualization primitives based on Matplotlib.

1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.decomposition import PCA


def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None]*2
    c = x*np.array([[0, 1., 0]]) + (2-x)*np.array([[1., 0, 0]])
    return np.clip(c, 0, 1)


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4/3] * n
    figsize = [sum(ratios)*4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors='lime', ps=4):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c in zip(axes, kpts, colors):
        a.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, indices=(0, 1), a=1.):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1, transform=fig.transFigure, c=color[i], linewidth=lw,
            alpha=a)
            for i in range(len(kpts0))]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches='tight', pad_inches=0, **kw)


def draw_keypoints(img, kpts, color=(0, 255, 0), radius=4, s=3):
    img = np.uint8(img)
    if s != 1:
        img = cv2.resize(img, None, fx=s, fy=s)
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, -1)
    for k in np.int32(kpts):
        cv2.circle(img, tuple(s*k), radius, color,
                   thickness=-1, lineType=cv2.LINE_AA)
    return img

def draw_matches(img1, kp1, img2, kp2, matches, color=None, kp_radius=5,
                 thickness=2, margin=20):
    # Create frame
    if len(img1.shape) == 2:
        img1 = img1[..., np.newaxis]
    if len(img2.shape) == 2:
        img2 = img2[..., np.newaxis]
    if img1.shape[-1] == 1:
        img1 = np.repeat(img1, 3, -1)
    if img2.shape[-1] == 1:
        img2 = np.repeat(img2, 3, -1)
    new_shape = (max(img1.shape[0], img2.shape[0]),
                 img1.shape[1]+img2.shape[1]+margin,
                 img1.shape[2])
    new_img = np.ones(new_shape, type(img1.flat[0]))*255

    # Place original images
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],
            img1.shape[1]+margin:img1.shape[1]+img2.shape[1]+margin] = img2

    # Draw lines between matches
    if not isinstance(color, list):
        color = [color]*len(matches)
    for m, c in zip(matches, color):
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not c:
            if len(img1.shape) == 3:
                c = np.random.randint(0, 256, 3)
            else:
                c = np.random.randint(0, 256)
            c = (int(c[0]), int(c[1]), int(c[2]))

        end1 = tuple(np.round(kp1[m[0]]).astype(int))
        end2 = tuple(np.round(kp2[m[1]]).astype(int)
                     + np.array([img1.shape[1]+margin, 0]))
        cv2.line(new_img, end1, end2, c, thickness, lineType=cv2.LINE_AA)
        cv2.circle(
            new_img, end1, kp_radius, c, thickness, lineType=cv2.LINE_AA)
        cv2.circle(
            new_img, end2, kp_radius, c, thickness, lineType=cv2.LINE_AA)
    return new_img


def draw_dense_descriptors(*maps):
    pca = PCA(n_components=3, svd_solver='full')
    dims = [m.shape[-1] for m in maps]
    assert len(np.unique(dims)) == 1
    dim = dims[0]
    all_maps = np.concatenate([m.reshape(-1, dim) for m in maps])
    pca.fit(all_maps)
    projected = [pca.transform(m.reshape(-1, dim)).reshape(m.shape[:2]+(3,))
                 for m in maps]
    _min = np.min([np.min(p) for p in projected])
    _max = np.max([np.max(p) for p in projected])
    return [(p - _min) / (_max - _min) for p in projected]