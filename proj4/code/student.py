import numpy as np
from skimage.feature import hog
from utils import *
import os.path as osp
from glob import glob
from random import shuffle
from sklearn.svm import LinearSVC

from turtle import distance
import numpy as np
import matplotlib
from skimage.io import imread
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.spatial.distance import cdist

def get_feature(file_list, template_size, cell_size, block_size):
    feats = []
    for i, file_name in enumerate(file_list):
        img = load_image_gray(file_name)
        img = resize(img, (template_size, template_size))
        # img = (img - img.mean()) / img.var()
        # if len(img.shape) == 3:
        #     img = img.mean(axis=-1)
        img_feature = hog(img, pixels_per_cell=(cell_size, cell_size), cells_per_block=(block_size, block_size))
        feats.append(img_feature)

    feats = np.array(feats)
#     print(feats.shape)
    return feats

def get_positive_features(train_path_pos, feature_params):
    """
    This function should return all positive training examples (faces) from
    36x36 images in 'train_path_pos'. Each face should be converted into a
    HoG template according to 'feature_params'.

    Useful functions:
    -   skimage.feature.hog(im, pixels_per_cell = (*, *)): computes HoG features
        eg:
                skimage.feature.hog(im, pixels_per_cell=(cell_size, cell_size),
                         cells_per_block=(n_cell, n_cell),
                         orientations=31)

        You can also use cyvlfeat, a vlfeat lib for python. 

    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    positive_files = glob(osp.join(train_path_pos, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
    feats = get_feature(positive_files, win_size, cell_size, 1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """
    This function should return negative training examples (non-faces) from any
    images in 'non_face_scn_path'. Images should be loaded in grayscale because
    the positive training data is only available in grayscale (use
    load_image_gray()).

    Useful functions:
    -   Useful functions:
    -   skimage.feature.hog(im, pixels_per_cell = (*, *)): computes HoG features
        eg:
                skimage.feature.hog(im, pixels_per_cell=(cell_size, cell_size),
                         cells_per_block=(n_cell, n_cell),
                         orientations=31)
    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.

    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
    feats = get_feature(negative_files, win_size, cell_size, 1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats

def train_classifier(features_pos, features_neg, C):
    """
    This function trains a linear SVM classifier on the positive and negative
    features obtained from the previous steps. We fit a model to the features
    and return the svm object.

    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """
    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
    svm = LinearSVC(C=C)
    train_data = np.concatenate([features_pos, features_neg], axis=0)
    train_label = []
    train_label.extend([1 for _ in range(features_pos.shape[0])])
    train_label.extend([0 for _ in range(features_neg.shape[0])])
#     svm.fit(features_pos, [1 for _ in range(features_pos.shape[0])])
#     svm.fit(features_neg, [0 for _ in range(features_neg.shape[0])])
    svm.fit(train_data, train_label)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm

def mine_hard_negs(non_face_scn_path, svm, feature_params):
    """
    This function is pretty similar to get_random_negative_features(). The only
    difference is that instead of returning all the extracted features, you only
    return the features with false-positive prediction.

    Useful functions:
    -   skimage.feature.hog(im, pixels_per_cell = (*, *)): computes HoG features
        eg:
                skimage.feature.hog(im, pixels_per_cell=(cell_size, cell_size),
                         cells_per_block=(n_cell, n_cell),
                         orientations=31)

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object

    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
#     feats = []
#     block_size = 1
#     for i, file_name in enumerate(negative_files):
#         img = imread(file_name)
#         img = resize(img, (win_size, win_size))
#         hog_feature = hog(img, pixels_per_cell=(cell_size, cell_size), cells_per_block=(block_size, block_size))
#         predict = svm.predict(np.array([hog_feature]))
#         if predict == 1:
#             feats.append(hog_feature)
    feats = get_feature(negative_files, win_size, cell_size, 1)
    predicts = svm.predict(feats)
    feats = feats[predicts == 1]

    feats = np.array(feats)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats

def run_detector(test_scn_path, svm, feature_params, verbose=False):
    """
    This function returns detections on all of the images in a given path. You
    will want to use non-maximum suppression on your detections or your
    performance will be poor (the evaluation counts a duplicate detection as
    wrong). The non-maximum suppression is done on a per-image basis. The
    starter code includes a call to a provided non-max suppression function.

    The placeholder version of this code will return random bounding boxes in
    each test image. It will even do non-maximum suppression on the random
    bounding boxes to give you an example of how to call the function.

    Your actual code should convert each test image to HoG feature space with
    a _single_ call to hog() for each scale. Then step over the HoG
    cells, taking groups of cells that are the same size as your learned
    template, and classifying them. If the classification is above some
    confidence, keep the detection and then pass all the detections for an
    image to non-maximum suppression. For your initial debugging, you can
    operate only at a single scale and you can skip calling non-maximum
    suppression. Err on the side of having a low confidence threshold (even
    less than zero) to achieve high enough recall.
    
    Useful functions:
    -   skimage.feature.hog(im, pixels_per_cell = (*, *)): computes HoG features
        eg:
                skimage.feature.hog(im, pixels_per_cell=(cell_size, cell_size),
                         cells_per_block=(n_cell, n_cell),
                         orientations=31)

    Args:
    -   test_scn_path: (string) This directory contains images which may or
            may not have faces in them. This function should work for the
            MIT+CMU test set but also for any other images (e.g. class photos).
    -   svm: A trained sklearn.svm.LinearSVC object
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slowerbecause the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time.
    -   verbose: prints out debug information if True

    Returns:
    -   bboxes: N x 4 numpy array. N is the number of detections.
            bboxes(i,:) is [x_min, y_min, x_max, y_max] for detection i.
    -   confidences: (N, ) size numpy array. confidences(i) is the real-valued
            confidence of detection i.
    -   image_ids: List with N elements. image_ids[i] is the image file name
            for detection i. (not the full path, just 'albert.jpg')
    """
    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    image_ids = []

    # number of top detections to feed to NMS
    topk = 40

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    scale_factor = feature_params.get('scale_factor', 0.65)
    template_size = int(win_size / cell_size)
    confidence_thr = 0.8

    bboxes = []
    confidences = []
    
    for idx, im_filename in enumerate(im_filenames):
        print(idx, '/', len(im_filenames), end='\r')
        # print('Detecting faces in {:s}'.format(im_filename), end='\r')
        im = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = im.shape
        
        #######################################################################
        #                        TODO: YOUR CODE HERE                         #
        #######################################################################
        # 1. create scale space HOG pyramid and return scores for prediction

        # 2. scale image. We suggest you create a scale factor list and use recurrence 
        #    to generate image feature. eg:
        #       multi_scale_factor = np.array([1.0, 0.7, 0.5, 0.3])
        #       for scala_rate in multi_scale_factor:
        #           scale img
        #           xxx
        multi_scale_factor = [1.0, 0.7, 0.5, 0.3]
        bbox_w = 36
        bbox_h = 36
        block_size = 1
        face_bbox = []
        for scale_id, scale_factor in enumerate(multi_scale_factor):
        #     print(im_shape)
            H, W = im_shape
            scale_img = resize(im, (int(H*scale_factor), int(W*scale_factor)))
            H, W = scale_img
        #     print(scale_img.shape)
            temp_conf_list = []
            temp_bboxes_list = []
            for i in range(0, H - bbox_h - 1, bbox_h):
                for j in range(0, W - bbox_w - 1, bbox_w):
                    temp_img = scale_img[i:i+bbox_h, j:j+bbox_w]
                #     print(temp_img.shape)
                    if temp_img.shape[0] == 0 or temp_img.shape[1] == 0:
                        continue
                    temp_img = resize(temp_img, (bbox_h, bbox_w))
                    img_feature = hog(temp_img, pixels_per_cell=(cell_size, cell_size), cells_per_block=(block_size, block_size))                    
                #     predict = svm.predict(np.array([img_feature]))
                    temp_confidence = svm.decision_function(np.array([img_feature]))
                #     print(temp_confidence.shape)
                #     print([predict, temp_confidence])
                #     print(temp_confidence)
                    if temp_confidence > confidence_thr:
                        confidences.append(temp_confidence)
                        image_ids.append(im_filename.split('/')[-1])
                        bboxes.append([i/scale_factor, j/scale_factor, (i + bbox_h)/scale_factor, (j + bbox_w)/scale_factor])
                        # temp_conf_list.append(temp_confidence)
                        # temp_bboxes_list.append([i/scale_factor, j/scale_factor, (i + bbox_h)/scale_factor, (j + bbox_w)/scale_factor])
                        # face_bbox.append([i/scale_factor, j/scale_factor, (i + bbox_h)/scale_factor, (j + bbox_w)/scale_factor])
        #     print(len(temp_conf_list))                        
        #     temp_conf_list = np.array(temp_conf_list)
            
        #     sort_temp_conf_list = np.argsort(temp_conf_list)[::-1][:]

        # print(face_bbox)
        # break 
    

        # 3. image to hog feature
        # 4. sliding windows at scaled feature map. you can use horizontally 
        #    recurrence and and vertically recurrence
        # 5. extract feature for current bounding box and use classify
        # 6. record your result for this image
        # 7. non-maximum suppression (NMS)
        #    non_max_supr_bbox() can actually get somewhat slow with thousands of
        #    initial detections. You could pre-filter the detections by confidence,
        #    e.g. a detection with confidence -1.1 will probably never be
        #    meaningful. You probably _don't_ want to threshold at 0.0, though. You
        #    can get higher recall with a lower threshold. You should not modify
        #    anything in non_max_supr_bbox(). If you want to try your own NMS methods,
        #    please create another function.
        #    eg:
        #     is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
        #        im_shape, verbose=verbose)






        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

        


#     return bboxes, confidences, image_ids
#     print(np.array(confidences).shape)           
    return np.array(bboxes), np.array(confidences).squeeze(), image_ids




if __name__ == '__main__':
    feature_params = {'template_size': 36, 'hog_cell_size': 6}
    svm = None
    data_path = osp.join('..','data')
    train_path_pos = osp.join(data_path, 'caltech_faces', 'Caltech_CropFaces')
    non_face_scn_path = osp.join(data_path, 'train_non_face_scenes')
    test_scn_path = osp.join(data_path, 'test_scenes', 'test_jpg')
    label_filename = osp.join(data_path, 'test_scenes', 'ground_truth_bboxes.txt')
    num_negative_examples = 10000
    features_pos = get_positive_features(train_path_pos, feature_params)
    print(features_pos.shape)
    features_neg = get_random_negative_features(non_face_scn_path, feature_params,
                                               num_negative_examples)
    svm = train_classifier(features_pos, features_neg, 5e-2)
    hard_negs = mine_hard_negs(non_face_scn_path, svm, feature_params)
    features_neg_2 = np.vstack((features_neg, hard_negs))
    svm_2 = train_classifier(features_pos, features_neg_2, 5e-2)
    print(hard_negs.shape)
    

    bboxes, confidences, image_ids = run_detector(test_scn_path, svm, feature_params, verbose=False)
    print(confidences.shape)
    cond = confidences.squeeze()
#     order = np.argsort(-cond)
#     print(order)
#     cond = cond[order]
    gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections = evaluate_detections(bboxes, cond,
                                                                                    image_ids, label_filename)
    print(image_ids)