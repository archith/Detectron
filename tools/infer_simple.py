#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import pandas as pd
from tqdm import tqdm

from caffe2.python import workspace

sys.path.append('lib')
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils


c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--image-prefix',
        dest='image_prefix',
        help='image file name prefix',
        default='',
        type=str
    )
    parser.add_argument(
        'im_or_folder',
        help='image or folder of images',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-csv-file',
        dest='output_csv_file',
        help='output csv file with detection files',
        default='output.csv',
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):

    if os.path.isdir(args.im_or_folder):
        im_list = glob.glob(os.path.join(args.im_or_folder, args.image_prefix + '*.' + args.image_ext))
    else:
        assert False, "Has to be a folder with images extracted from images"

    im_list.sort()

    if os.path.isfile(args.output_csv_file):
        print('CSV file already present : {}'.format(args.output_csv_file))
        return

    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    video_name = os.path.basename(args.im_or_folder)



    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    det_bbox_df = pd.DataFrame(columns=[u'video_name', u'image_name', u'box_ymin', u'box_xmin',
       u'box_ymax', u'box_xmax', u'box_score', u'box_class', u'img_width', u'img_height'])

    video_name_list = []
    image_name_list = []
    box_ymin_list = []
    box_xmin_list = []
    box_ymax_list = []
    box_xmax_list = []
    box_score_list = []
    class_list = []
    img_width_list = []
    img_height_list = []

    logger.info('Processing {} -> {}'.format(args.im_or_folder, args.output_dir))
    for im_path in tqdm(im_list, desc='{:25}'.format(os.path.basename(args.im_or_folder))):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_path))
        )
        #logger.info('Processing {} -> {}'.format(im_path, out_name))
        im = cv2.imread(im_path)
        im_h, im_w, _ = im.shape
        im_name = os.path.basename(im_path)

        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        # logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        # for k, v in timers.items():
        #     logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        # if i == 0:
        #     logger.info(
        #         ' \ Note: inference on the first image will be slower than the '
        #         'rest (caches and auto-tuning need to warm up)'
        #     )

        person_boxes = cls_boxes[1]

        # Ignore all classes other than person for visualization
        viz_boxes = [[] for x in range(len(cls_boxes))]
        viz_boxes[1] = person_boxes

        for bbox in person_boxes:
            xmin, ymin, xmax, ymax, score = bbox

            video_name_list.append(video_name)
            image_name_list.append(im_name)

            box_xmin_list.append(xmin/im_w)
            box_ymin_list.append(ymin/im_h)
            box_xmax_list.append(xmax/im_w)
            box_ymax_list.append(ymax/im_h)

            box_score_list.append(score)

            class_list.append('person')

            img_width_list.append(im_w)
            img_height_list.append(im_h)


        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            viz_boxes,
            segms=None, #cls_segms,
            keypoints=None, #cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2,
            ext='jpg'
        )

        pass

    det_bbox_df['video_name'] = video_name_list
    det_bbox_df['image_name'] = image_name_list
    det_bbox_df['box_ymin'] = box_ymin_list
    det_bbox_df['box_xmin'] = box_xmin_list
    det_bbox_df['box_ymax'] = box_ymax_list
    det_bbox_df['box_xmax'] = box_xmax_list
    det_bbox_df['box_score'] = box_score_list
    det_bbox_df['box_class'] = class_list
    det_bbox_df['img_width'] = img_width_list
    det_bbox_df['img_height'] = img_height_list

    det_bbox_df.to_csv(args.output_csv_file)
    print('Write results to {}'.format(args.output_csv_file))

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
