import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle

from .AY20 import AY20
from .AY21 import AY21

def get_last_pid(dataset):
    last = 0
    for d in dataset.query:
        _, pid, _, _, _ = d
        last = pid if last < pid else last
    return last

class AY_Test(BaseImageDataset):

    def __init__(self, root='', verbose=True):
        super(AY_Test, self).__init__()
        
        ay20 = AY20(root=root, pid_begin=0)
        # ay20과 pid 겹치지 않도록 ay20의 마지막 pid를 offset으로 설정
        last_pid = get_last_pid(ay20)
        ay21 = AY21(root=root, pid_begin=last_pid)
        
        train = ay20.train + ay21.train
        query = ay20.query + ay21.query
        gallery = ay20.gallery + ay21.gallery

        if verbose:
            print("=> AY_test loaded")
            self.print_dataset_statistics(train, query, gallery)
        

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

        print()
