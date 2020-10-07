import numpy as np
import cv2


def normalize(l, axis=-1):
    return np.array(l) / np.linalg.norm(l, axis=axis, keepdims=True)


def root_descriptors(d, axis=-1):
    return np.sqrt(d / np.sum(d, axis=axis, keepdims=True))


def matching(desc1, desc2, do_ratio_test=False, cross_check=True):
    if desc1.dtype == np.bool and desc2.dtype == np.bool:
        desc1, desc2 = np.packbits(desc1, axis=1), np.packbits(desc2, axis=1)
        norm = cv2.NORM_HAMMING
    else:
        desc1, desc2 = np.float32(desc1), np.float32(desc2)
        norm = cv2.NORM_L2

    if do_ratio_test:
        matches = []
        matcher = cv2.BFMatcher(norm)
        for m, n in matcher.knnMatch(desc1, desc2, k=2):
            m.distance = 1.0 if (n.distance == 0) else m.distance / n.distance
            matches.append(m)
    else:
        matcher = cv2.BFMatcher(norm, crossCheck=cross_check)
        matches = matcher.match(desc1, desc2)
    return matches_cv2np(matches)


def fast_matching(desc1, desc2, ratio_thresh, labels=None):
    '''A fast matching method that matches multiple descriptors simultaneously.
       Assumes that descriptors are normalized and can run on GPU if available.
       Performs the landmark-aware ratio test if labels are provided.
    '''
    import torch
    cuda = torch.cuda.is_available()

    desc1, desc2 = torch.from_numpy(desc1), torch.from_numpy(desc2)
    if cuda:
        desc1, desc2 = desc1.cuda(), desc2.cuda()

    with torch.no_grad():
        dist = 2*(1 - desc1 @ desc2.t())
        dist_nn, ind = dist.topk(2, dim=-1, largest=False)
        match_ok = (dist_nn[:, 0] <= (ratio_thresh**2)*dist_nn[:, 1])

        if labels is not None:
            labels = torch.from_numpy(labels)
            if cuda:
                labels = labels.cuda()
            labels_nn = labels[ind]
            match_ok |= (labels_nn[:, 0] == labels_nn[:, 1])

        if match_ok.any():
            matches = torch.stack(
                [torch.nonzero(match_ok)[:, 0], ind[match_ok][:, 0]], dim=-1)
        else:
            matches = ind.new_empty((0, 2))

    return matches.cpu().numpy()


def topk_matching(query, database, k):
    '''Retrieve top k matches from a database (shape N x dim) with a single
       query. In order to reduce any overhead, use numpy instead of PyTorch
    '''
    dist = 2 * (1 - database @ query)
    ind = np.argpartition(dist, k)[:k]
    ind = ind[np.argsort(dist[ind])]
    return ind


def matches_cv2np(matches_cv):
    matches_np = np.int32([[m.queryIdx, m.trainIdx] for m in matches_cv])
    distances = np.float32([m.distance for m in matches_cv])
    return matches_np.reshape(-1, 2), distances
