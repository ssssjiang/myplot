import matplotlib.pyplot as plt
from matplotlib import cm
import random
import cv2
import numpy as np

from .model_utils.read_write_model import read_update_images_text, read_points3D_text
from myplot_tools.colmap_helper.visiable_utils.viz import plot_images, plot_keypoints


def read_image(path):
    assert path.exists(), path
    image = cv2.imread(str(path))
    if len(image.shape) == 3:
        image = image[:, :, ::-1]
    return image


def visualize_sfm_2d(sfm_model, image_dir, color_by='visibility',
                     selected=[], n=1, seed=0, dpi=75):
    assert sfm_model.exists()
    assert image_dir.exists()

    # images = read_images_binary(sfm_model / 'images.bin')
    images = read_update_images_text(sfm_model / 'images.txt')
    if color_by in ['track_length', 'depth']:
        # points3D = read_points3d_binary(sfm_model / 'points3D.bin')
        points3D = read_points3D_text(sfm_model / 'points3D.txt')

    if not selected:
        image_ids = list(images.keys())
        selected = random.Random(seed).sample(image_ids, n)

    for i in selected:
        name = images[i].name
        image = read_image(image_dir / name)
        keypoints = images[i].xys
        visible = images[i].point3D_ids != -1

        if color_by == 'visibility':
            color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
        elif color_by == 'track_length':
            tl = np.array([len(points3D[j].image_ids) if j != -1 else 1
                           for j in images[i].point3D_ids])
            max_, med_ = np.max(tl), np.median(tl[tl > 1])
            tl = np.log(tl)
            color = cm.jet(tl / tl.max()).tolist()
            text = f'max/median track length: {max_}/{med_}'
        elif color_by == 'depth':
            p3ids = images[i].point3D_ids
            p3D = np.array([points3D[j].xyz for j in p3ids if j != -1])
            z = (images[i].qvec2rotmat() @ p3D.T)[-1] + images[i].tvec[-1]
            z -= z.min()
            color = cm.jet(z / np.percentile(z, 99.9))
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
            keypoints = keypoints[visible]
        else:
            raise NotImplementedError(f'Coloring not implemented: {color_by}.')

        plot_images([image], dpi=dpi)
        plot_keypoints([keypoints], colors=[color], ps=4)
        fig = plt.gcf()
        fig.text(
            0.01, 0.99, text, transform=fig.axes[0].transAxes,
            fontsize=10, va='top', ha='left', color='k',
            bbox=dict(fc=(1, 1, 1, 0.5), edgecolor=(0, 0, 0, 0)))
        fig.text(
            0.01, 0.01, name, transform=fig.axes[0].transAxes,
            fontsize=5, va='bottom', ha='left', color='w')
        fig.show()