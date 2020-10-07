import sys, os
import numpy as np
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation as R, Slerp
from shutil import copyfile
import argparse


def getImgList(imgPath, code):
    if not os.path.exists(imgPath):
        print('image path not exits')
        exit()
    else:
        list_img = [f for f in os.listdir(imgPath) if code in f]
        list_img.sort(key=lambda name: int(name.split('.')[0]))
        return [os.path.join(imgPath, f) for f in list_img]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_dataset', required=True)
    parser.add_argument('--vertices_path', required=True)
    parser.add_argument('--hfnet_ws', required=True)
    parser.add_argument('--sample', required=True)

    args = parser.parse_args()
    return args


def getFovParam(dataset_path):
    camParamFile = os.path.join(dataset_path, 'cameraParam.txt')
    try:
        content = open(camParamFile).readlines()
        block = []
        paramsDic = {}

        for i, line in enumerate(content):
            if "fisheye instrinsic" in line:
                print(line)
                print('i:', i)
                block = content[i: i + 6]
                break

        paramsDic['fx'] = float(block[2].split(' ')[0])
        paramsDic['fy'] = float(block[3].split(' ')[1])
        paramsDic['cx'] = float(block[2].split(' ')[2])
        paramsDic['cy'] = float(block[3].split(' ')[2])
        paramsDic['w'] = float(block[5].split(' ')[-1])
        print('find fov param:', paramsDic)
        return paramsDic
    except IOError:
        print("cameraParam.txt doesn't exist please check:{}".format(camParamFile))


def main():
    args = parse_args()

    vertices_path = args.vertices_path
    imgDir = os.path.join(args.mapping_dataset, 'fisheye')
    imgTime = os.path.join(args.mapping_dataset, 'fisheye_timestamps.txt')
    sfmModelPath = os.path.join(args.hfnet_ws, 'initial_model')
    outputDir = os.path.join(args.hfnet_ws, 'im')

    if not os.path.isdir(sfmModelPath):
        os.makedirs(sfmModelPath)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    sfmImagesTxt = open(os.path.join(sfmModelPath, 'images.txt'), 'w')
    sfmCamerasTxt = open(os.path.join(sfmModelPath, 'cameras.txt'), 'w')
    sfmPoints3DTxt = open(os.path.join(sfmModelPath, 'points3D.txt'), 'w')

    imgList = getImgList(imgDir, '.jpg')
    tmp = np.loadtxt(vertices_path, dtype=np.str, delimiter=',')
    tmp = tmp[1:]
    viMap = []
    imgTime = np.loadtxt(imgTime, usecols=(2,)) * 1000

    cam_map = getFovParam(args.mapping_dataset)
    sfmCamerasTxt.write("1 FOV 640 480 {} {} {} {} {}".format(cam_map['fx'], cam_map['fy'],
                                                              cam_map['cx'], cam_map['cy'],
                                                              cam_map['w']))

    sfmCamerasTxt.close()
    sfmPoints3DTxt.close()

    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    else:
        filelist = os.listdir(outputDir)
        for img in filelist:
            os.remove("{}/{}".format(outputDir, img))

    selectedIdx = []
    selectedPose = []
    for frame in tmp:
        viMap.append([float(a) for a in frame])

    sample = 1
    for v in viMap:
        idx = np.argmin(np.abs(v[1] - imgTime))
        sample += 1
        if np.abs(imgTime[idx] - v[1]) < 50000000 and sample % int(args.sample) == 0:
            selectedIdx.append(idx)
            selectedPose.append(v)

    for i in range(len(selectedPose)):
        p = selectedPose[i]
        T_w_c = np.eye(4)
        T_w_c[:3, 3] = np.array([p[2], p[3], p[4]])
        T_w_c[:3, :3] = (R.from_quat([p[5], p[6], p[7], p[8]])).as_dcm()
        T_c_w = np.linalg.inv(T_w_c)
        quat = (R.from_dcm(T_c_w[:3, :3])).as_quat()
        t = T_c_w[:3, 3]
        sfmImagesTxt.write("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {} {:0>5}.png \n\n".
                           format(i + 1, quat[3], quat[0], quat[1], quat[2], t[0], t[1], t[2], 1, i + 1))
        copyfile(imgList[selectedIdx[i]], outputDir + '/' + '{:0>5}.png'.format(i + 1))

    sfmImagesTxt.close()


if __name__ == '__main__':
    main()
