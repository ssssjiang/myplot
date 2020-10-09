import sys
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
from pathlib import Path
from myplot_tools.settings import EXPER_PATH, DATA_PATH
import argparse
import os
import cv2

sys.path.append(sys.path[0] + '/../')


class HFNet:
    def __init__(self, model_path, outputs):
        self.session = tf.Session()
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))

        net_input = tf.image.rgb_to_grayscale(self.image_ph[None])
        tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], str(model_path),
            clear_devices=True,
            input_map={'image:0': net_input})

        graph = tf.get_default_graph()
        self.outputs = {n: graph.get_tensor_by_name(n + ':0')[0] for n in outputs}
        self.nms_radius_op = graph.get_tensor_by_name('pred/simple_nms/radius:0')
        self.num_keypoints_op = graph.get_tensor_by_name('pred/top_k_keypoints/k:0')

    def inference(self, image, nms_radius=4, num_keypoints=1000):
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        return self.session.run(self.outputs, feed_dict=inputs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_image_dir', required=True)
    parser.add_argument('--out_npz_dir', required=True)
    parser.add_argument('--num_keypoints', type=int, default=1000)

    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--crop', action='store_true')
    args = parser.parse_args()
    return args


model_path = Path(EXPER_PATH, 'saved_models/hfnet')
outputs = ['global_descriptor', 'keypoints', 'local_descriptors', 'scores']
hfnet = HFNet(model_path, outputs)

args = parse_args()
db_path = args.in_image_dir
db_list = os.listdir(db_path)
print(db_list)
db_list.sort()

db_image = lambda n: cv2.imread(db_path + n)[:, :, ::-1]

export_path = args.out_npz_dir
if not os.path.isdir(export_path):
    os.mkdir(export_path)
if ~args.crop and args.mask:
    mask = cv2.imread(export_path + "../mask.png", 0)

    for i in range(len(db_list)):
        data = hfnet.inference(db_image(db_list[i]), num_keypoints=args.num_keypoints)

        filtered = []
        mask_count = 0
        for k in data['keypoints']:
            if mask[k[1], k[0]] != 0:
                filtered.append(mask_count)
            mask_count += 1

        export = {
            'keypoints': data['keypoints'][filtered],
            'local_descriptors': data['local_descriptors'][filtered],
            'global_descriptor': data['global_descriptor'],
            'scores': data['scores']
        }
        name = export_path + db_list[i][:-4]
        np.savez(f'{name}.npz', **export)
        print("save", i, name)

else:
    sty = 0;
    stx = 40;
    height = 240;
    width = 560;
    endy = sty + height;
    endx = stx + width
    print("crop  x:", stx, "-", endx, "y:", sty, "-", endy)
    for i in range(len(db_list)):
        img = db_image(db_list[i])
        if args.crop:
            img = img[sty:endy, stx:endx]
        data = hfnet.inference(img, num_keypoints=args.num_keypoints)
        if args.crop:
            for ii in range(len(data['keypoints'])):
                data['keypoints'][ii][0] += stx
                data['keypoints'][ii][1] += sty

        export = {
            'keypoints': data['keypoints'],
            'local_descriptors': data['local_descriptors'],
            'global_descriptor': data['global_descriptor'],
            'scores': data['scores']
        }
        name = export_path + db_list[i][:-4]
        np.savez(f'{name}.npz', **export)
        print("save", i, name)
