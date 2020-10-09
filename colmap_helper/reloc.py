import sys
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
from pathlib import Path
from myplot_tools.settings import EXPER_PATH, DATA_PATH
import argparse
import os
import cv2
from timeit import default_timer
from myplot_tools.colmap_helper.visiable_utils.utils import plot_images, plot_matches, add_frame
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

sys.path.append(sys.path[0] + '/../')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--printmatchratio', action='store_true')
    parser.add_argument('--use_filtered_db', action='store_true')
    parser.add_argument('--use_querynpz', action='store_true')
    parser.add_argument('--num_keypoints', type=int, default=1000)
    parser.add_argument('--modelPath', required=True)
    parser.add_argument('--npz_dir', required=True)
    parser.add_argument('--result_path', required=True)
    parser.add_argument('--db_im_path', required=True)
    parser.add_argument('--query_path', required=True)
    parser.add_argument('--sample', type=int, required=True)
    parser.add_argument('--query_data_set', required=True)
    parser.add_argument('--unique3dpoints', type=int, required=True)
    parser.add_argument('--nn', type=int, required=True)
    parser.add_argument('--ratio_test', type=float, required=True)

    # parser.add_argument('--mask', action='store_true')
    args = parser.parse_args()
    return args


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

    def inference(self, image, nms_radius=0, num_keypoints=1000):
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            # self.image_ph: image.astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        return self.session.run(self.outputs, feed_dict=inputs)


def compute_distance(desc1, desc2):
    # For normalized descriptors, computing the distance is a simple matrix multiplication.
    return 2 * (1 - desc1 @ desc2.T)


def kpData2Dic(data):
    assert len(data) % 3 == 0
    dict = {}
    for i in range(0, len(data), 3):
        dict[int(i / 3)] = [float(data[i]), float(data[i + 1]), int(data[i + 2])]
    return dict


def loadSfmModel(filePath):
    p = open(filePath + '/points3D.txt')
    tmp3d = [line.split(' ') for line in p.readlines()[3:]]
    points3D = {}
    for line in tmp3d:
        e = [float(e) for e in line]
        points3D[e[0]] = e[1:]
    i = open(filePath + '/images.txt')
    images = {}
    imagesData = i.readlines()[4:]
    for i in range(0, len(imagesData), 2):
        l1 = imagesData[i].split(' ')
        l2 = imagesData[i + 1].split(' ')
        if len(l1) == 10:
            kpDic = kpData2Dic(l2)
            name = l1[-1][:-1]
            pose = [float(e) for e in l1[1:8]]
            images[name] = {'pose': pose, 'keypointsDic': kpDic}
    return points3D, images


def getFovParam(dataset_path):
    camParamFile = os.path.join(dataset_path, 'cameraParam.txt')
    try:
        content = open(camParamFile).readlines()
        block = []
        paramsDic = {}
        for i, line in enumerate(content):
            if "fisheye intrinsic" in line:
                block = content[i:i + 6]
                break
        paramsDic['fx'] = float(block[2].split(' ')[0])
        paramsDic['fy'] = float(block[3].split(' ')[1])
        paramsDic['cx'] = float(block[2].split(' ')[2])
        paramsDic['cy'] = float(block[3].split(' ')[2])
        paramsDic['w'] = float(block[5].split(' ')[-1])
        print("find fov param :", paramsDic)
        return paramsDic
    except IOError:
        print("cameraParam.txt doesn't exist please check:{}".format(camParamFile))


def undistortionFOV(dps, fx, fy, cx, cy, w):
    ups = dps.copy()
    #     print("dp0:{} dp1:{} fx:{}".format(d_p[0],d_p[1],fx)
    i = 0
    for row in dps:
        rowu = undistortionFOV1pt(row, fx, fy, cx, cy, w)
        ups[i, :] = rowu
        i += 1
    return ups


def undistortionFOV1pt(distP, fx, fy, cx, cy, w):
    u_p = [0., 0.]
    d_p = [0., 0.]
    #     print("dp0:{} dp1:{} fx:{}".format(d_p[0],d_p[1],fx)
    d_p[0] = (float(distP[0]) - cx) / fx
    d_p[1] = (float(distP[1]) - cy) / fy
    #     print("dp0:{} dp1:{} fx:{}".format((float(d_p[0]) - cx)/fx,float(d_p[1]) - cx,fx))
    mul2tanwby2 = np.tan(w / 2.0) * 2.0

    #     Calculate distance from point to center.
    r_d = np.sqrt(d_p[0] * d_p[0] + d_p[1] * d_p[1])
    if mul2tanwby2 == 0 or r_d == 0:
        print("tanw error")
        return u_p

    #     Calculate undistorted radius of point.
    kMaxValidAngle = 89.0
    if abs(r_d * w) <= kMaxValidAngle:
        r_u = np.tan(r_d * w) / (r_d * mul2tanwby2)
    else:
        print('angle not valid')
        return u_p

    u_p[0] = d_p[0] * r_u
    u_p[1] = d_p[1] * r_u

    u_p[0] = u_p[0] * fx + cx
    u_p[1] = u_p[1] * fy + cy
    return u_p


def match_with_ratio_test(desc1, desc2, thresh):
    t0 = default_timer()
    dist = compute_distance(desc1, desc2)
    if args.verbose:
        print("MWR CD", desc1.shape, desc2.shape, " in %0.3fs" % (default_timer() - t0))  # 1000*256
    t0 = default_timer()
    NEAREST = np.argpartition(dist, 2, axis=-1)[:, :2]
    # print(nearest)
    if args.verbose: print("MWR argpart", dist.shape, " in %0.3fs" % (default_timer() - t0))
    # t0=default_timer()
    dist_nearest = np.take_along_axis(dist, NEAREST, axis=-1)
    # print("MWR take_along_axis", nearest.shape," in %0.3fs"%(default_timer()-t0))#1000*2
    # t0=default_timer()
    valid_mask = dist_nearest[:, 0] <= (thresh ** 2) * dist_nearest[:, 1]
    # print("MWR mask", dist_nearest.shape," in %0.3fs"%(default_timer()-t0))#1000*2
    # t0=default_timer()
    matches = np.stack([np.where(valid_mask)[0], NEAREST[valid_mask][:, 0]], 1)
    if args.printmatchratio: print("matchratio ", dist_nearest.shape, matches.shape)
    # print("MWR stack", nearest.shape, valid_mask.shape," in %0.3fs"%(default_timer()-t0))#1000
    return matches


model_path = Path(EXPER_PATH, 'saved_models/hfnet')
outputs = ['global_descriptor', 'keypoints', 'local_descriptors', 'scores']
hfnet = HFNet(model_path, outputs)

print("input paths must have / at end")
args = parse_args()
plotMatch = args.plot

cameraParam = getFovParam(args.query_data_set)

fx = cameraParam['fx']
fy = cameraParam['fy']
cx = cameraParam['cx']
cy = cameraParam['cy']
w = cameraParam['w']

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

modelPath = args.modelPath
match_result_path = os.path.join(args.result_path, 'matches')
if not os.path.isdir(match_result_path):
    os.makedirs(match_result_path)

result_images = open(os.path.join(args.result_path, 'images.txt'), 'w')
result_reloc = open(os.path.join(args.result_path, 'reloc1-1.txt'), 'w')

db_im_path = args.db_im_path
query_path = args.query_path

db_list = os.listdir(db_im_path)
query_list = os.listdir(query_path)
db_list = [f for f in db_list if '.png' in f or '.jpg' in f]
query_list = [f for f in query_list if '.png' in f or '.jpg' in f]
query_list.sort(key=lambda name: int(name.split('.')[0]))
db_list.sort(key=lambda name: int(name.split('.')[0]))

if args.use_filtered_db:
    goodimagestxt = open(model_path + '../goodimages.txt')
    db_list = [line for line in goodimagestxt.read().splitlines()]
    db_list.sort()
    print(db_list)

db_image = lambda n: cv2.imread(os.path.join(db_im_path, n))[:, :, ::-1]
query_image = lambda n: cv2.imread(os.path.join(query_path, n))[:, :, ::-1]

if plotMatch:
    t0 = default_timer()
    images_db = [db_image(frame) for frame in db_list]
    print("cvread ", len(db_list), " db images in %0.3fs" % (default_timer() - t0))

t0 = default_timer()

images_query = [query_image(frame) for frame in query_list]
print("cvread ", len(query_list), " query images in %0.3fs" % (default_timer() - t0))

t0 = default_timer()
npz_dir = args.npz_dir  # EXPER_PATH + '/exports/sfm/db_grass/'
dbimageprenames = [imagefullname[:-4] for imagefullname in db_list]
db = [np.load(os.path.join(npz_dir, dbimageprename + '.npz')) for dbimageprename in dbimageprenames]
global_index = np.stack([d['global_descriptor'] for d in db])
# print(global_index)
print("numpyload ", len(db_list), " db npz in %0.3fs" % (default_timer() - t0))

# Localization Process

if not os.path.isdir(match_result_path):
    os.makedirs(match_result_path)
else:
    filelist = os.listdir(match_result_path)
    for img in filelist:
        os.remove("{}/{}".format(match_result_path, img))

dbPoints3D, dbImages = loadSfmModel(modelPath)

count = 0
# knn candidate
nn = args.nn
ratio_test = args.ratio_test

relocResult = []
locResultStr = []
contestResultStr = []
keypoints2D = {}
print("ratio_test: ", ratio_test, "nn: ", nn, "num_keypoints: ", args.num_keypoints)
print("modelPath: ", modelPath)
print("npz_dir: ", npz_dir)
print("db_images: ", db_im_path)
print("query_images: ", query_path)
print("db_size: ", len(db_list), " query_size: ", len(query_list))
query_npz_dir = npz_dir + "../querynpz/"
count_test = 0
success_img = []
fail_img = []
for i in range(0, len(query_list), args.sample):
    count_test += 1
    # for i in range(0,50,4):
    tloop0 = default_timer()

    print('\n', count, "------infering------", query_list[i])
    t0 = default_timer()
    globalKpNum = []
    query = {}
    if args.use_querynpz:
        try:
            query = np.load(query_npz_dir + query_list[i][:-4] + '.npz')
        except:
            continue
        else:
            pass
    else:
        query = hfnet.inference(images_query[i], num_keypoints=args.num_keypoints)
    print("infered in %0.3fs" % (default_timer() - t0))
    t0 = default_timer()
    nearest = np.argsort(compute_distance(query['global_descriptor'], global_index))[:nn]
    undistKps = np.array([[0., 0.]], dtype=np.float32)
    imps = np.array([[0., 0., 0.]])
    allMatches = np.array([[0, 0]])
    if args.verbose: print("infered1 ", i)
    if args.verbose: print("after in %0.3fs" % (default_timer() - t0))

    tmatchall0 = default_timer()
    for item in nearest:
        t0 = default_timer()
        matches = match_with_ratio_test(query['local_descriptors'],
                                        db[item]['local_descriptors'], ratio_test)
        if args.verbose: print("match one in %0.3fs" % (default_timer() - t0))
        if len(matches) <= 3:
            continue

        t0 = default_timer()
        distKp = query['keypoints'][matches[:, 0]]
        # print("undist 0",distKp.shape," in %0.3fs" % (default_timer() - t0))
        # t0=default_timer()
        distKp = np.array([distKp], dtype=np.float32)
        distKp = distKp.reshape((-1, 2))
        # print("undist 1",distKp.shape," in %0.3fs" % (default_timer() - t0))
        # print(distKp)
        # t0 = default_timer()
        undistKp = undistortionFOV(distKp, fx, fy, cx, cy, w)
        undistKp = undistKp.reshape((-1, 2))
        if args.verbose: print("undistkp 1 ", undistKp)
        if args.verbose: print("undist 2", undistKp.shape, " in %0.3fs" % (default_timer() - t0))

        if db_list[item] in dbImages:
            impsIdx = [dbImages[db_list[item]]['keypointsDic'][m[1]][2] for m in matches]
        else:
            continue
        t0 = default_timer()
        impsIdx = np.array(impsIdx)
        a = np.where(np.array(impsIdx) >= 0)
        if len(a) < 1:
            continue
        undistKp = undistKp[a]
        impsIdx = impsIdx[a]
        matches = matches[a]
        if args.verbose: print("filter one in %0.3fs" % (default_timer() - t0))

        t0 = default_timer()
        imp = [dbPoints3D[idx][:3] for idx in impsIdx]
        imp = np.array(imp)
        if args.verbose: print("get3D one in %0.3fs" % (default_timer() - t0))

        if args.verbose: print("undistkp 2 ", undistKp)

        if args.verbose: print("imp ", imp)

        if len(imp) < 1:
            continue

        t0 = default_timer()
        imps = np.concatenate((imps, imp), axis=0)
        undistKps = np.concatenate((undistKps, undistKp), axis=0)
        allMatches = np.concatenate((allMatches, matches), axis=0)
        globalKpNum.append([item, len(imps) - 1])
        if args.verbose: print("append one in %0.3fs" % (default_timer() - t0))
    if args.verbose: print("infered2 ", i)
    print(nn, " matched ", [db_list[item] for item in nearest[:3]], " in %0.3fs" % (default_timer() - tmatchall0))

    imps = imps[1:]
    undistKps = undistKps[1:]
    allMatches = allMatches[1:]

    t0 = default_timer()
    if len(imps) <= 4:
        print("FAIL len imps ", imps)
        continue
    success, R_vec, t, inliers = cv2.solvePnPRansac(
        imps, undistKps, K, np.array([0., 0., 0., 0.]),
        iterationsCount=5000, reprojectionError=5.0,  # confidence=0.99,#no use
        flags=cv2.SOLVEPNP_P3P)
    # check for unique elements number
    uniqueKps = undistKps[inliers]
    if args.verbose: print("uniqueKps ", uniqueKps)
    uniqueDic = {}
    for ui in uniqueKps:
        if not ui[0][0] in uniqueDic:
            uniqueDic[ui[0][0]] = ui

    if not success or len(inliers) < 10 or len(uniqueDic) <= args.unique3dpoints:  # 10
        fail_img.append(query_list[i])

    if len(uniqueDic) <= args.unique3dpoints:  # 10
        print("FAIL uniqueDic ", len(uniqueDic))
        continue

    print("success ", success, "len(inliers)", len(inliers), " uniqueDic ", len(uniqueDic))

    if success and len(inliers) >= 10:  # 10
        success_img.append(query_list[i])
        inliers = inliers[:, 0]
        num_inliers = len(inliers)
        #         inlier_ratio = len(inliers) / len(undistKp)
        success &= num_inliers >= 5

        ret, R_vec, t = cv2.solvePnP(
            imps[inliers], undistKps[inliers], K,
            np.array([0., 0., 0., 0.]), rvec=R_vec, tvec=t,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        assert ret

        print("trans: [ %0.2f %0.2f %0.2f ]" % (t[0, 0], t[1, 0], t[2, 0]))
        T_query_w = np.eye(4)
        T_query_w[:3, :3] = cv2.Rodrigues(R_vec)[0]
        T_query_w[:3, 3] = t[:, 0]
        q = (R.from_dcm(T_query_w[:3, :3])).as_quat()
        t = T_query_w[:3, 3]
        ## for iros workshop
        T_w_query = np.linalg.inv(T_query_w)
        q1 = (R.from_dcm(T_w_query[:3, :3])).as_quat()
        t1 = T_w_query[:3, 3]

        print("pnp in %0.3fs" % (default_timer() - t0))

        locResultStr.append("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} 1 {}\n \n"
                            .format(count, q[3], q[0], q[1], q[2], t[0], t[1], t[2], query_list[i]))
        contestResultStr.append("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {}\n"
                                .format(q1[0], q1[1], q1[2], q1[3], t1[0], t1[1], t1[2], query_list[i]))
        matches = allMatches[inliers]

        if plotMatch == True:
            print("saving figs at ", match_result_path)
            t0 = default_timer()
            for j in range(len(allMatches)):
                if j in inliers:
                    continue
                else:
                    allMatches[j][1] = -1
            for ii in range(len(globalKpNum)):

                if ii == 0:
                    m = allMatches[:globalKpNum[ii][1]]
                    m = m[m[:, 1] >= 0]
                    plot_matches(images_query[i], query['keypoints'],
                                 images_db[globalKpNum[ii][0]], db[globalKpNum[ii][0]]['keypoints'],
                                 m, color=(0, 1, 0), dpi=100, thickness=0.5, kp_size=6)
                else:
                    m = allMatches[globalKpNum[ii - 1][1]:globalKpNum[ii][1]]
                    m = m[m[:, 1] >= 0]
                    plot_matches(images_query[i], query['keypoints'],
                                 images_db[globalKpNum[ii][0]], db[globalKpNum[ii][0]]['keypoints'],
                                 m, color=(0, 1, 0), dpi=100, thickness=0.5, kp_size=6)
                plt.savefig(os.path.join(match_result_path, str(query_list[i]) + "_matches" + str(ii) + ".png"))
                plt.cla()
                plt.close('all')

            allMatches = allMatches[inliers]
            if len(nearest) > 1:
                plot_images([images_query[i]], keypoints=[query['keypoints'][allMatches[:, 0]]])
                plt.savefig(os.path.join(match_result_path, str(query_list[i]) + "_matches.png"))
                plt.cla()
                plt.close('all')

            print("plot in %0.3fs" % (default_timer() - t0))

    count += 1
    print("relocalization happend with:{} total reloc number:{}/{}".format(query_list[i], count, count_test),
          " in %0.3fs" % (default_timer() - tloop0))

print("failed img: ", fail_img)
print("successful img: ", success_img)

for l in locResultStr:
    result_images.write(l)
result_images.close()

for l in contestResultStr:
    result_reloc.write(l)
result_reloc.close()

print("success:{} fail:{} ratio:{}".format(len(success_img), len(fail_img),
                                           float(len(success_img) / (len(fail_img) + len(success_img)))))