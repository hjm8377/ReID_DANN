# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle

class Market_NightReID(BaseImageDataset):
    
    dataset_dir = 'market1501'
    dataset_n_dir = 'NightReID'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Market_NightReID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)


        self.train_n_dir = osp.join(self.dataset_n_dir, 'bounding_box_train')
        self.query_n_dir = osp.join(self.dataset_n_dir, 'query')
        self.gallery_n_dir = osp.join(self.dataset_n_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin + self.num_train_pids
        train_n = self._process_n_dir(self.train_n_dir, relabel=True)
        query_n = self._process_n_dir(self.query_n_dir, relabel=False)
        gallery_n = self._process_n_dir(self.gallery_n_dir, relabel=False)

        if verbose:
            print("=> NightReID loaded")
            self.print_dataset_statistics(train_n, query_n, gallery_n)

        self.train_n = train_n
        self.query_n = query_n
        self.gallery_n = gallery_n

        self.num_train_n_pids, self.num_train_n_imgs, self.num_train_n_cams, self.num_train_n_vids = self.get_imagedata_info(self.train_n)
        self.num_query_n_pids, self.num_query_n_imgs, self.num_query_n_cams, self.num_query_n_vids = self.get_imagedata_info(self.query_n)
        self.num_gallery_n_pids, self.num_gallery_n_imgs, self.num_gallery_n_cams, self.num_gallery_n_vids = self.get_imagedata_info(self.gallery_n)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _process_n_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'(\d{4})([LR][123])C')
        camdic = {
            'L1': 0, 'L2': 1, 'L3': 2,
            'R1': 3, 'R2': 4, 'R3': 5,
        }

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = pattern.search(img_path).groups()
            pid_container.add(int(pid))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = pattern.search(img_path).groups()
            pid=int(pid)
            camid = camdic[camid]
            assert 1 <= pid <= 7474
            assert 0 <= camid <= 5
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, pid, camid, 1))
        return dataset