import numpy as np
import cv2
from collections import namedtuple


LocResult = namedtuple(
    'LocResult', ['success', 'num_inliers', 'inlier_ratio', 'T'])
loc_failure = LocResult(False, 0, 0, None)


def do_pnp(kpts, lms, query_info, config):
    kpts = kpts.astype(np.float32).reshape((-1, 1, 2))
    lms = lms.astype(np.float32).reshape((-1, 1, 3))

    success, R_vec, t, inliers = cv2.solvePnPRansac(
        lms, kpts, query_info.K, np.array([query_info.dist, 0, 0, 0]),
        iterationsCount=5000, reprojectionError=config['reproj_error'],
        flags=cv2.SOLVEPNP_P3P)

    if success:
        inliers = inliers[:, 0]
        num_inliers = len(inliers)
        inlier_ratio = len(inliers) / len(kpts)
        success &= num_inliers >= config['min_inliers']

        ret, R_vec, t = cv2.solvePnP(
                lms[inliers], kpts[inliers], query_info.K,
                np.array([query_info.dist, 0, 0, 0]), rvec=R_vec, tvec=t,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        assert ret

        query_T_w = np.eye(4)
        query_T_w[:3, :3] = cv2.Rodrigues(R_vec)[0]
        query_T_w[:3, 3] = t[:, 0]
        w_T_query = np.linalg.inv(query_T_w)

        ret = LocResult(success, num_inliers, inlier_ratio, w_T_query)
    else:
        inliers = np.empty((0,), np.int32)
        ret = loc_failure

    return ret, inliers
