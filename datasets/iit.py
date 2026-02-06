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
class IIT(BaseImageDataset):

    dataset_dir = 'BoxTrack-ReID'

    def __init__(self, root='', verbose=True, pid_begin = 0, syn_prob=0, **kwargs):
        self.syn_prob=syn_prob
        super(IIT, self).__init__()
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
            print("=> IIT_ReID Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

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
        if self.syn_prob == 0:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            pattern = re.compile(r'([-\d]+)_C(\d+)')

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
                # assert 0 <= pid <= 1501  # pid == 0 means background
                # assert 1 <= camid <= 6
                camid -= 1  # index starts from 0
                if relabel: 
                    pid = pid2label[pid]
                dataset.append((img_path, self.pid_begin + pid, camid, 1, 1))
            return dataset
        
        else:
            import random
            syn_folders = []
            origin_folder = 'market1501'

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
                # assert 0 <= pid <= 1501  # pid == 0 means background
                # assert 1 <= camid <= 6
                camid -= 1  # index starts from 0
                if relabel: 
                    pid = pid2label[pid]
                
                # syn_prob 이상이면 syn폴더로 대체
                if random.uniform(0, 1) >= self.syn_prob:
                    syn_folder = random.choice(syn_folders)
                    img_path = img_path.replace(origin_folder, syn_folder)
                    dataset.append((img_path, self.pid_begin + pid, camid, 1))
                else:
                    dataset.append((img_path, self.pid_begin + pid, camid, 1))
            return dataset