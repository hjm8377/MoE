import itertools
from .bases import BaseImageDataset
from utils.logger import setup_logger
import logging

# 공개 주간 데이터
from .market1501 import Market1501
from .msmt17 import MSMT17
from .duke import DukeMTMC
from .cuhk03 import CUHK03

# 공개 야간 데이터
from .nightreid import NightReID
from .night600 import Night600

# 비공개 주간 데이터
from .anyang import Anyang
from .KIST import KIST
from .DataTang import DataTang

# 비공개 야간 데이터
from .iit import IIT
from .datatang import DataTang_IR, DataTang_RGB, DataTang_Night



_factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'dukemtmc': DukeMTMC,
    'cuhk03': CUHK03,

    'nightreid': NightReID,
    'night600': Night600,

    'anyang': Anyang,
    'kist': KIST,
    'datatang': DataTang,

    'iit': IIT,
    'datatang_ir': DataTang_IR,
    'datatang_rgb': DataTang_RGB,
    'datatang_night': DataTang_Night,
}

class ReID(BaseImageDataset):
    def __init__(self, root='', verbose=False, **kwargs):
        super(ReID, self).__init__()
        
        cfg = kwargs.get('cfg', None)
        self.dataset_dir = root

        self.logger = logging.getLogger("transreid.check")

        self.dataset_list = cfg.DATASETS.SPECS
        if self.dataset_list is None or len(self.dataset_list) == 0:
            raise ValueError("DATASETS.SPECS is empty in config file.")
        
        self.train = []
        self.query = []
        self.gallery = []

        self.test_name = ''

        self.all_pids_num = 0
        self.all_cams_num = 0
        self._draw_hline(self.logger)
        self.logger.info(f"==== Training datasets ====")
        for dataset in self.dataset_list:
            name, use_test_for_training, use_for_testing = dataset.split(':')

            data = _factory[name](root=self.dataset_dir, verbose=False)

            train_data = data.train
            query_data = data.query
            gallery_data = data.gallery

            relabeled_train_data, len_train_pids, len_train_cams = self._relabel_pid(train_data, pid_offset=self.all_pids_num, cam_offset=self.all_cams_num)
            self.all_pids_num += len_train_pids
            self.all_cams_num += len_train_cams

            if use_test_for_training == 'True':
                relabeled_test_data, len_test_pids, len_test_cams = self._relabel_pid(query_data + gallery_data, pid_offset=self.all_pids_num, cam_offset=self.all_cams_num-len_train_cams)
                self.all_pids_num += len_test_pids
                self.all_cams_num += len_test_cams

                train_data = relabeled_train_data + relabeled_test_data
                
                self.logger.info(f"[{name:^12}] | #   train ids: {len_train_pids + len_test_pids:6,d} | #   train images: {len(relabeled_train_data) + len(relabeled_test_data):10,d}")

            else:
                train_data = relabeled_train_data
                self.logger.info(f"[{name:^12}] | #   train ids: {len_train_pids:6,d} | #   train images: {len(relabeled_train_data):10,d}")

            self.train.extend(train_data)

            # Testset은 한 개의 데이터셋만 사용한다고 가정
            if use_for_testing == 'True':
                self.query.extend(query_data)
                self.gallery.extend(gallery_data)
                self.test_name += f"{name}"


        self._draw_hline(self.logger)
        self.logger.info(f"[Total Train ] | #   train ids: {len(set(pid for _, pid, _, _, _ in self.train)):6,d} | #   train images: {len(self.train):10,d}")
        self._draw_hline(self.logger)
        self.logger.info(f"==== Testing datasets ====")
        self.logger.info(f"[{self.test_name:^12}] | #   query ids: {len(set(pid for _, pid, _, _, _ in self.query)):6,d} | #   query images: {len(self.query):10,d}")
        self.logger.info(f"               | # gallery ids: {len(set(pid for _, pid, _, _, _ in self.gallery)):6,d} | # gallery images: {len(self.gallery):10,d}")
        self._draw_hline(self.logger)
        self.logger.info("\n")


        if verbose:
            self.logger.info("=> ReID combined dataset loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        assert len(self.train) > 0, "Training set is empty"
        assert len(self.query) > 0, "Query set is empty"
        assert len(self.gallery) > 0, "Gallery set is empty"

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

        # print(self.num_train_pids, self.num_train_imgs, self.num_train_cams)

    def _relabel_pid(self, data, pid_offset, cam_offset):
        if not data:
            return data, 0, 0
        
        uniq_pids = sorted({pid for _, pid, _, _, _ in data})
        uniq_cams = sorted({cam for _, _, cam, _, _ in data})

        pid_map = {old: pid_offset + i for i, old in enumerate(uniq_pids)}
        cam_map = {old: cam_offset + i for i, old in enumerate(uniq_cams)}

        new_items = []
        for path, pid, cam, view, domain in data:
            new_items.append((path, pid_map[pid], cam_map[cam], view, domain))

        return new_items, len(uniq_pids), len(uniq_cams)
    
    def _draw_hline(self, logger):
        logger.info(f"---------------------------------------------------------------------")