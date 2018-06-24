from collections import defaultdict
import numpy as np
import os
import json
from glob import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from dataset.object_data import ObjectData
from utils.dataset_util import rotate, overlay_image_alpha


class AVAretail(ObjectData):
    def __init__(self, cfg):
        self.cfg = cfg
        bkg_images = glob(os.path.join(cfg.bkg_data_dir, "*"))
        self.bkg_images = [b for b in bkg_images if b.lower().endswith('.jpg')
                           or b.lower().endswith('.jpeg')]
        product_dirs = glob(os.path.join(cfg.obj_data_dir, "*"))
        self.product_names = {i + 1: os.path.basename(p)
                              for i, p in enumerate(product_dirs)}
        self.images = {}
        for i, prod_dir in enumerate(product_dirs):
            prod_images = glob(os.path.join(product_dirs[i], "*"))
            prod_images = [p for p in prod_images if p.lower().endswith('.png')]
            self.images[i + 1] = prod_images
        self.n_products = len(self.images)
        self.shuffled_labels = None
        self.reset()
        self.save_train_data()
        annotation_files = glob(os.path.join(
            self.cfg.train_data_dir, "*.json"))
        super().__init__(cfg, annotation_files)

    def reset(self):
        idx = 1 + np.arange(self.n_products)
        np.random.shuffle(idx)
        self.shuffled_labels = idx

    def generator(self):
        n1 = self.cfg.n_items_min
        n2 = self.cfg.n_items_max
        count = 0
        while True:
            label = self.shuffled_labels[count]
            n = np.random.randint(n1, n2 + 1)
            images = np.random.choice(self.images[label], n)
            count += 1
            if count >= self.n_products:
                self.reset()
                count = 0
            yield self.generate_image(images, label), label

    def generate_image(self, images, label):
        img_h, img_w = self.cfg.image_shape[0], self.cfg.image_shape[1]
        bkg_img = np.random.choice(self.bkg_images)
        image = plt.imread(bkg_img)
        h, w, _ = image.shape
        scale = max(1.5 * img_h / h, 1.5 * img_w / w)
        scale = np.random.uniform(scale, 3. * scale)
        image = cv2.resize(image, None, fx=scale, fy=scale)
        h, w, _ = image.shape
        top = np.random.randint(0, h - img_h)
        left = np.random.randint(0, w - img_w)
        image = image[top: top + img_h, left: left + img_w]
        flip_h, flip_v = np.random.randint(0, 1, 2)
        if flip_h:
            image = cv2.flip(image, 0)
        if flip_v:
            image = cv2.flip(image, 1)
        if np.max(image) > 1.:
            image = image / 255.
        # color = np.ones((img_h, img_w, 3)) * np.random.uniform(0, 1., 3)
        # image = cv2.addWeighted(image, 0.7, color, 0.3, 0)
        img_h, img_w, _ = image.shape
        for img_file in images:
            img = plt.imread(img_file)
            mask = img[:, :, 3]
            dims = self.cfg.product_dims[
                self.cfg.category[self.product_names[label]]]
            y_idx, x_idx = np.where(mask > 0.5)
            y1, x1 = np.min(y_idx), np.min(x_idx)
            y2, x2 = np.max(y_idx), np.max(x_idx)
            img = img[y1:y2, x1:x2, :]
            h, w, _ = img.shape
            volume = dims[2] * dims[0] * dims[1]
            scale = img_h / np.random.uniform(10, 12)
            scale *= (volume / (h * w * w)) ** (1. / 3)
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
            img = rotate(img, np.random.uniform(-180, 180))
            mask = img[:, :, 3]
            h, w, _ = img.shape
            # col = i % ncols
            # row = i // ncols
            # delta_x, delta_y = np.random.uniform(0.2, 0.8, 2)
            # x = (delta_x + col) * img_w / ncols - 0.5 * w
            # y = (delta_y + row) * img_h / nrows - 0.5 * h
            y_idx, x_idx = np.where(mask > 0.9)
            y1, x1 = np.min(y_idx), np.min(x_idx)
            y2, x2 = np.max(y_idx), np.max(x_idx)
            img = img[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]
            h, w = y2 - y1, x2 - x1
            x = np.random.uniform(-0.2 * w, img_w - .75 * w)
            y = np.random.uniform(-0.2 * h, img_h - .75 * h)
            # classes.append(label)
            # bboxes.append([(y + y1) / img_h, (x + x1) / img_w,
            #                (y + y2) / img_h, (x + x2) / img_w])
            pos = (int(x), int(y))
            image = overlay_image_alpha(image, img[:, :, :3], pos, mask)
            # if np.random.uniform() < 0.15:
            #     x1, y1 = 0., np.random.uniform()
            #     x2, y2 = 1., np.random.uniform()
            #     x1, y1 = int(x1 * img_w), int(y1 * img_h)
            #     x2, y2 = int(x2 * img_w), int(y2 * img_h)
            #     mask = np.zeros((img_h, img_w))
            #     thickness = np.random.randint(5, 25)
            #     mask = cv2.line(mask, (x1, y1), (x2, y2), (1.), thickness)
            #     foreground_img = np.random.choice(self.bkg_images)
            #     foreground_img = plt.imread(foreground_img)
            #     foreground_img = foreground_img[:img_h, :img_w]
            #     color = np.ones((img_h, img_w, 3)) * np.random.uniform(0, 1., 3)
            #     if np.max(foreground_img) > 1.5:
            #         foreground_img = foreground_img / 255.
            #     foreground_img = cv2.addWeighted(foreground_img, 0.3, color, 0.7, 0)
            #     image = overlay_image_alpha(image, foreground_img, (0, 0), mask)
        #                 color = np.random.uniform(0., 1., 3)
        #                 thickness = np.random.randint(5, 25)
        #                 image = cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        #             if np.random.uniform() < 0.1:
        #                 hand_idx = np.random.randint(len(hand_images))
        #                 hand_img_file = hand_images[hand_idx]
        #                 img = plt.imread(hand_img_file)
        #                 h, w, _ = img.shape
        #                 scale = img_h / max(h, w)
        #                 if np.max(img) > 1.001:
        #                     img = img / 255.
        #                 if not hand_img_file.lower().endswith('png'):
        #                     new_img = np.zeros((h, w, 4))
        #                     new_img[:, :, :3] = img
        #                     new_img[:, :, 3] = np.prod(img, axis=2) < .9
        #                     img = new_img
        #                 img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        #                 img = rotate(img, np.random.uniform(-180, 180))
        #                 mask = img[:, :, 3]
        #                 h, w, _ = img.shape
        #                 x, y = np.random.uniform(-0.2, 0.2, 2)
        #                 pos = (int(x * img_w), int(y * img_h))
        #                 image = overlay_image_alpha(image, img[:, :, :3], pos, mask)
        blur = 2 * np.random.randint(0, 3) + 1
        image = cv2.GaussianBlur(image, (blur, blur), 0)
        image = np.clip(image, 0., 1.)
        return image.astype(np.float32)

    def save_train_data(self):
        data_dir = self.cfg.train_data_dir
        data_name = self.cfg.data_name
        img_dir = os.path.join(data_dir, data_name)
        labels_file = os.path.join(data_dir, data_name + '.json')
        if os.path.exists(labels_file):
            return
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        gen = self.generator()
        annotations = []
        for i in tqdm(range(self.cfg.n_samples)):
            filename = os.path.join(data_name, str(i).zfill(7) + '.jpg')
            filepath = os.path.join(data_dir, filename)
            image, label = next(gen)
            plt.imsave(filepath, image)
            annotations.append({u'filename': filename,
                                u'label': int(label)})
        j = json.dumps(annotations, indent=4)
        with open(labels_file, 'w') as f:
            f.write(j)

    def _build_dataset(self, dataset):
        for i, annotations in tqdm(enumerate(dataset)):
            img_id = i
            img_name = annotations['filename']
            label = annotations['label']
            h, w = self.cfg.image_shape
            self.imgs[img_id] = {'filename': img_name,
                                 'shape': [h, w],
                                 'label': label}

    def create_index(self):
        # create index
        print('creating index...')
        self.imgs, self.anns = {}, defaultdict(list)

        for dataset in self.datasets:
            self._build_dataset(dataset)

        print('index created!')
        self.ids = list(self.imgs.keys())
