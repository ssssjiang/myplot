import argparse
import os
import sqlite3
import sys
import numpy as np
from timeit import default_timer
import cv2

IS_PYTHON3 = sys.version_info[0] >= 3


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def compute_distance(desc1, desc2):
    # For normalized descriptors, computing the distance is a simple matrix multiplication.
    # return 2 * (1 - desc1 @ desc2.T)
    return 2 * (1 - np.dot(desc1, desc2.T))


def match_frames(path_npz1, path_npz2, num_points,
                 use_ratio_test, ratio_test_values):
    t0 = default_timer()
    frame1 = np.load(path_npz1)
    frame2 = np.load(path_npz2)
    print("load in %0.3fs" % (default_timer() - t0))

    # WARNING: scores are not taken into account as of now.
    des1 = frame1['local_descriptors'].astype('float32')[:num_points]
    des2 = frame2['local_descriptors'].astype('float32')[:num_points]

    if use_ratio_test:
        keypoint_matches = [[] for _ in ratio_test_values]
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(des1, des2, k=2)

        smallest_distances = [dict() for _ in ratio_test_values]

        matches_mask = [[0, 0] for _ in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            for ratio_idx, ratio in enumerate(ratio_test_values):
                if m.distance < ratio * n.distance:
                    if m.trainIdx not in smallest_distances[ratio_idx]:
                        smallest_distances[ratio_idx][m.trainIdx] = (
                            m.distance, m.queryIdx)
                        matches_mask[i] = [1, 0]
                        keypoint_matches[ratio_idx].append(
                            (m.queryIdx, m.trainIdx))
                    else:
                        old_dist, old_queryIdx = smallest_distances[
                            ratio_idx][m.trainIdx]
                        if m.distance < old_dist:
                            old_distance, old_queryIdx = smallest_distances[
                                ratio_idx][m.trainIdx]
                            smallest_distances[ratio_idx][m.trainIdx] = (
                                m.distance, m.queryIdx)
                            matches_mask[i] = [1, 0]
                            keypoint_matches[ratio_idx].remove(
                                (old_queryIdx, m.trainIdx))
                            keypoint_matches[ratio_idx].append(
                                (m.queryIdx, m.trainIdx))
    else:
        keypoint_matches = [[]]
        matches_mask = []
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)

        # Matches are already cross-checked.
        for match in matches:
            # match.trainIdx belongs to des2.
            keypoint_matches[0].append((match.queryIdx, match.trainIdx))

    return keypoint_matches


def match_frames_noload(frame1, frame2, num_points,
                 use_ratio_test, ratio_test_values):
    # t0=default_timer()
    # print("load in %0.3fs" % (default_timer() - t0))
    # Assert the keypoints are sorted according to the score.
   # assert np.all(np.sort(frame1['scores'])[::-1] == frame1['scores'])

    # WARNING: scores are not taken into account as of now.
    des1 = frame1['local_descriptors'].astype('float32')[:num_points]
    des2 = frame2['local_descriptors'].astype('float32')[:num_points]

    if use_ratio_test:
        keypoint_matches = [[] for _ in ratio_test_values]
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(des1, des2, k=2)

        smallest_distances = [dict() for _ in ratio_test_values]

        matches_mask = [[0, 0] for _ in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            for ratio_idx, ratio in enumerate(ratio_test_values):
                if m.distance < ratio * n.distance:
                    if m.trainIdx not in smallest_distances[ratio_idx]:
                        smallest_distances[ratio_idx][m.trainIdx] = (
                            m.distance, m.queryIdx)
                        matches_mask[i] = [1, 0]
                        keypoint_matches[ratio_idx].append(
                            (m.queryIdx, m.trainIdx))
                    else:
                        old_dist, old_queryIdx = smallest_distances[
                            ratio_idx][m.trainIdx]
                        if m.distance < old_dist:
                            old_distance, old_queryIdx = smallest_distances[
                                ratio_idx][m.trainIdx]
                            smallest_distances[ratio_idx][m.trainIdx] = (
                                m.distance, m.queryIdx)
                            matches_mask[i] = [1, 0]
                            keypoint_matches[ratio_idx].remove(
                                (old_queryIdx, m.trainIdx))
                            keypoint_matches[ratio_idx].append(
                                (m.queryIdx, m.trainIdx))
    else:
        keypoint_matches = [[]]
        matches_mask = []
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)

        # Matches are already cross-checked.
        for match in matches:
            # match.trainIdx belongs to des2.
            keypoint_matches[0].append((match.queryIdx, match.trainIdx))

    return keypoint_matches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_out_path', required=True)
    parser.add_argument('--min_num_matches', type=int, default=15)
    parser.add_argument('--max_num_global_neighbor', type=int, default=50)
    parser.add_argument('--num_points_per_frame', type=int, default=2500)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--npz_dir', required=True)

    parser.add_argument('--thresh_loadnpz', type=int, default=1000)

    parser.add_argument('--use_ratio_test', action='store_true')
    parser.add_argument('--ratio_test_values', type=str, default='0.85')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    db_list = os.listdir(args.image_dir)
    db_list.sort()
    num_db = len(db_list)
    connection = sqlite3.connect(args.npz_dir + '../test.db')
    print('using test db')
    cursor = connection.cursor()

    def helper(name):
        cursor.execute('''select data from global_descriptor where name = ?;''', [name])
        dataglobal = cursor.fetchone()
        return np.array(blob_to_array(dataglobal[0], np.float32)).reshape(-1, 4096)[0]

    global_index = np.stack([helper(name) for name in db_list]).reshape(-1, 4096)
    print(global_index.shape)
    print(global_index[3, :])

    ratio_test_values = [float(v) for v in args.ratio_test_values.split(',')]
    print('Ratio test values to use:', ratio_test_values)
    outfiles = [open(args.match_out_path+'matches{}.txt'.format(x), 'w+')
                for x in [int(i * 100) for i in ratio_test_values]]

    matching_image_pairs = []
    setpairs = set()
    print('Looking for matching image pairs...')
    t00 = default_timer()
    npzstore = {}
    for i in range(num_db):
        curname = db_list[i]
        global_desc = global_index[i].copy()
        nearest = np.argsort(compute_distance(global_desc, global_index))[:args.max_num_global_neighbor]
        if num_db < args.thresh_loadnpz:
            npzstore[curname] = np.load(os.path.join(args.npz_dir, os.path.splitext(curname)[0] + '.npz'))
        for item in nearest:
            if item == i:
                continue
            if item < i and (item, i) in setpairs:
                continue
            matchname = db_list[item]
            matching_image_pairs.append((curname, matchname))
            print("(", curname, matchname, ")")
            fr = min(i, item)
            bk = max(i, item)
            setpairs.add((fr, bk))
    print('Got', len(matching_image_pairs), 'matching image pairs.', matching_image_pairs,
          " in %0.3fs " % (default_timer() - t00))

    num_missing_images = 0
    for (name1, name2) in matching_image_pairs:
        npz1 = os.path.join(args.npz_dir, os.path.splitext(name1)[0] + '.npz')
        npz2 = os.path.join(args.npz_dir, os.path.splitext(name2)[0] + '.npz')

        assert os.path.isfile(npz1), npz1
        assert os.path.isfile(npz2), npz2

        num_points = args.num_points_per_frame
        t0 = default_timer()

        if num_db < args.thresh_loadnpz:
            matches_for_different_ratios = match_frames_noload(
                npzstore[name1], npzstore[name2], num_points,
                args.use_ratio_test, ratio_test_values)
        else:
            matches_for_different_ratios = match_frames(
                npz1, npz2, num_points,
                args.use_ratio_test, ratio_test_values)

        print(name1, name2, " in %0.3fs with %d matches" % (default_timer() - t0, len(matches_for_different_ratios[0])))

        if (args.use_ratio_test):
            assert len(matches_for_different_ratios) == len(ratio_test_values)

        for i, keypoint_matches in enumerate(matches_for_different_ratios):
            if len(keypoint_matches) > args.min_num_matches:
                outfiles[i].write(name1 + ' ' + name2 + '\n')
                for (match1, match2) in keypoint_matches:
                    outfiles[i].write(str(match1) + ' ' + str(match2) + '\n')
                outfiles[i].write('\n')

    for outfile in outfiles:
        outfile.close()

    print('Missing', num_missing_images, 'images skipped.')


if __name__ == '__main__':
    main()
