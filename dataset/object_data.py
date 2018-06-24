import time
import json
import numpy as np
import os
from abc import abstractmethod
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from glob import glob
import cv2
from tqdm import tqdm
from utils import tfrecord_util
import tensorflow as tf
from utils.dataset_util import rotate, overlay_image_alpha


class ObjectData(object):
    def __init__(self, cfg, annotation_files):
        """
        Constructor of ObjectData class
        """
        self.cfg = cfg
        self.imgs, self.ids, self.anns = None, None, None
        self.data_dir = cfg.train_data_dir
        if annotation_files is not None:
            print('loading annotations into memory...')
            tic = time.time()
            self.datasets = []
            if type(annotation_files) != list:
                annotation_files = [annotation_files]
            for ann_file in annotation_files:
                dataset = json.load(open(ann_file, 'r'))
                self.datasets.append(dataset)
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.create_index()

    @abstractmethod
    def create_index(self):
        return

    def get_size(self):
        return len(self.ids)

    def _create_tf_example(self, img_id):
        img_meta = self.imgs[img_id]
        img_file = img_meta['filename']
        img_file = os.path.join(self.data_dir, img_file)
        img_shape = list(img_meta['shape'])
        label = img_meta['label']

        feature_dict = {
            'image/filename':
                tfrecord_util.bytes_feature(img_file.encode('utf8')),
            'image/shape':
                tfrecord_util.int64_list_feature(img_shape),
            'image/label':
                tfrecord_util.int64_feature(label)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def create_tf_record(self, out_path, shuffle=True):
        print("Creating tf records : ", out_path)
        writer = tf.python_io.TFRecordWriter(out_path)
        if shuffle:
            np.random.shuffle(self.ids)
        for img_id in tqdm(self.ids):
            tf_example = self._create_tf_example(img_id)
            writer.write(tf_example.SerializeToString())
        writer.close()
