import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from skimage.filters import _gaussian


def get_interest_points(image, feature_width, alpha=0.1, c_thr=0.5, num_thr=10):
    '''
    Returns a set of interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here!

    H, W = image.shape
    dx = get_x_gradient(image)
    dy = get_y_gradient(image)
    assert dx.shape == image.shape
    assert dy.shape == image.shape

    dxx = get_x_gradient(dx)
    dyy = get_y_gradient(dy)
    dxy = get_y_gradient(dx)
    assert dxx.shape == image.shape
    assert dyy.shape == image.shape
    assert dxy.shape == image.shape

    scores = dxx * dyy - dxy ** 2 - alpha * (dxx + dyy) ** 2
    
    # print(scores.min())
    # print(scores.mean())
    # ys, xs = np.where(scores >= c_thr)
    # score = scores[ys, xs]
    # print(scores)
    # print(score)
    # print(ys)
    # print(xs)
    sort_index = np.argsort(scores.reshape(-1))[::-1]
    ys, xs = sort_index // W, sort_index % W
    score = scores[ys, xs]
    thr_index = (score >= c_thr)
    ys = ys[thr_index]
    xs = xs[thr_index]
    score = score[thr_index]
    assert score.shape == ys.shape

    dist_x = xs[:, None] - xs[None, :]
    dist_y = ys[:, None] - ys[None, :]
    dist = (dist_x ** 2 + dist_y ** 2)

    ans_ys = []
    ans_xs = []
    for i in range(dist.shape[0]):
        if dist[i, i] == 0:
            ans_ys.append(ys[i])
            ans_xs.append(xs[i])
        else:
            continue
        for j in range(i, dist.shape[1]):
            if dist[i, j] < num_thr:
                dist[j][j] = -1


    ys = np.array(ans_ys)
    xs = np.array(ans_xs)

    print(len(ys))
    # dt = np.dtype([('x', int), ('y',int), ('score', int)])
    # a = np.array([(1,2,3), (2, 3, 4), (5,3, 1)])
    # np.sort(a, order='score')
    
    # point = np.stack([ys, xs], axis=0).transpose(1, 0)
    # temp = ((point[None, :, :] - point[:, None, :])**2).sum(axis=-1)
    # print(temp)
    # print(temp.shape)
    # print(temp.mean())
    
    # sort_index = np.argsort(scores)


    # These are placeholders - replace with the coordinates of your interest points!
    # xs = np.asarray([0])
    # ys = np.asarray([0])

    return xs, ys

def get_x_gradient(input):
    dx = input[1:, :] - input[:-1, :]
    dx = np.concatenate([dx, np.zeros((1, dx.shape[1]))], axis=0)
    
    return dx

def get_y_gradient(input):
    dy = input[:, 1:] - input[:, :-1]
    dy = np.concatenate([dy, np.zeros((dy.shape[0], 1))], axis=1)

    return dy


def get_features(image, x, y, feature_width):
    '''
    Returns a set of feature descriptors for a given set of interest points.

    (Please note that we reccomend implementing this function after you have implemented
    match_features)

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each descriptor_window_image_width/4.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! 

    # This is a placeholder - replace this with your features!
    import skimage

    features = []
    image = skimage.filters.gaussian(image, sigma=1)
    h, w = image.shape
    for i in range(len(x)):
        feature = np.zeros((16, 8))
        xi, yi = x[i], y[i]
        base = np.zeros((18, 18))
        xmin, xmax = int(max(0, yi-9)), int(min(h, yi+9))
        ymin, ymax = int(max(0, xi-9)), int(min(w, xi+9))
        # print(base[0:xmax-xmin, 0:ymax-ymin].shape)
        # print(h, w, xi)
        # print(ymin, ymax)
        # print(image[xmin:xmax, ymin:ymax].shape)
        base[0:xmax-xmin, 0:ymax-ymin] += image[xmin:xmax, ymin:ymax]
        dx = (base[2:, 1:-1] - base[1:-1, 1:-1])
        dy = (base[1:-1, 2:] - base[1:-1, 1:-1])
        
        for j in range(16):
            for k in range(16):
                index = to_index(dx[j][k], dy[j][k])
                feature[j//4 * 4 + k//4][index] += 1

        feature = feature.reshape(128)
        feature = (feature - feature.mean()) / feature.var()
        features.append(feature)


    # features = np.asarray([0])
    features = np.stack(features, axis=0)
    assert features.shape == (len(x), 128)

    return features


def to_index(dx, dy):
    flag = [
        [0, 6],
        [2, 4]
    ]

    x = 0 if dx >= 0 else 1
    y = 0 if dy >= 0 else 1
    base_index = flag[x][y]
    if x == y and np.abs(dx) >= np.abs(dy):
        base_index += 1
    elif x != y and np.abs(dx) < np.abs(dy):
        base_index += 1

    return base_index

def match_features(im1_features, im2_features, dist_thr = 0.1, ratio_thr = 0.8):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - zip (python built in function)

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here!

    # These are placeholders - replace with your matches and confidences!

    f1 = im1_features[:, None, :]
    f2 = im2_features[None, :, :]
    dist = np.sqrt(np.sum((f1 - f2) ** 2, axis=-1))
    assert dist.shape == (f1.shape[0], f2.shape[1])

    
    sort_dist = np.sort(dist)
    sort_index = np.argmin(dist, axis=1)
    print(sort_index)
    ratio = sort_dist[:, 0] / sort_dist[:, 1]
    f1_in = (sort_dist[:, 0] < dist_thr)
    f2_in = (ratio < ratio_thr)
    index = (f1_in & f2_in)
    print(np.where(index == True))
    # print(ratio)
    index = np.where(index == True)[0]
    print(index)
    print(sort_index[index].shape)
    

    matches = np.stack([index, sort_index[index]], axis=1)
    confidences = 1 - ratio[index]

    # matches = np.asarray([0,0])
    # confidences = np.asarray([0])

    return matches, confidences


if __name__ == '__main__':
    from skimage.transform import rescale
    from helpers import cheat_interest_points, evaluate_correspondence
    from utils import load_data
    from skimage.color import rgb2gray

    data_pair = "notre_dame"
    image1, image2, eval_file = load_data(data_pair)
    image1 = rgb2gray(image1)
    image2 = rgb2gray(image2)

    # make images smaller to speed up the algorithm. This parameter
    # gets passed into the evaluation code, so don't resize the images
    # except for changing this parameter - We will evaluate your code using
    # scale_factor = 0.5, so be aware of this

    scale_factor = 0.5
    # scale_factor = [0.5, 0.5, 1]

    # Bilinear rescaling
    image1 = np.float32(rescale(image1, scale_factor))
    image2 = np.float32(rescale(image2, scale_factor))

    feature_width = 16
    (x1, y1, x2, y2) = cheat_interest_points(eval_file, scale_factor)
    # print(x1.shape)
    # x1 = x1.reshape(-1)
    # x1.sort()
    # print(x1)
    # print(x1)
    # print(y1)
    # c_thr = 0.13
    c_thr = 0.01
    num_thr = 50
    (x1, y1) = get_interest_points(image1, feature_width, c_thr=c_thr, num_thr=num_thr)
    # (x2, y2) = get_interest_points(image2, feature_width, c_thr=c_thr)
    
    # print(x2)
    # print(y2)
    
    # image1_features = get_features(image1, x1, y1, feature_width)
    # image2_features = get_features(image2, x2, y2, feature_width)
    # matches, confidences = match_features(image1_features, image2_features, dist_thr=100, ratio_thr=0.9)
    # print(matches)