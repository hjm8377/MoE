from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import torch
import random
import numpy as np


class DomainRandomIdentitySampler(Sampler):
    """
    주/야간 데이터를 1:1 비율로 샘플링하는 RandomIdentitySampler

    Args:
    - data_source (list): list of (img_path, pid, camid, domain_label).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.num_day_pids = self.num_pids_per_batch // 2
        self.num_night_pids = self.num_pids_per_batch - self.num_day_pids

        self.day_index_dic = defaultdict(list)
        self.night_index_dic = defaultdict(list)

        for index, item in enumerate(self.data_source):
            pid = item[1]
            domain = item[4]
            if domain == 1:
                self.night_index_dic[pid].append(index)
            else:
                self.day_index_dic[pid].append(index)

        self.day_pids = list(self.day_index_dic.keys())
        self.night_pids = list(self.night_index_dic.keys())

        if len(self.day_pids) < self.num_day_pids or len(self.night_pids) < self.num_night_pids:
            raise ValueError("특정 도메인의 ID 개수가 배치당 필요한 개수보다 적습니다.")

        day_total_sets = sum(len(self.day_index_dic[pid]) // self.num_instances for pid in self.day_pids)
        night_total_sets = sum(len(self.night_index_dic[pid]) // self.num_instances for pid in self.night_pids)

        day_batches = day_total_sets // self.num_day_pids
        night_batches = night_total_sets // self.num_night_pids
        
        self.num_batches = max(day_batches, night_batches)
        self.length = self.num_batches * self.batch_size
        
        print(f"Sampler 초기화: Day Sets({day_total_sets}), Night Sets({night_total_sets})")
        print(f"최종 설정: {self.num_batches} Iters per Epoch (Total: {self.length})")

    def _prepare_batch_indices(self, index_dic, pids):
        """도메인별 PID-K 리스트 준비"""
        batch_idxs_dict = defaultdict(list)
        for pid in pids:
            idxs = copy.deepcopy(index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)

            for i in range(len(idxs) // self.num_instances):
                batch_idxs_dict[pid].append(idxs[i * self.num_instances : (i + 1) * self.num_instances])
        return batch_idxs_dict

    def __iter__(self):
        day_dict = self._prepare_batch_indices(self.day_index_dic, self.day_pids)
        night_dict = self._prepare_batch_indices(self.night_index_dic, self.night_pids)
        
        avai_day_pids = list(day_dict.keys())
        avai_night_pids = list(night_dict.keys())

        final_idxs = []

        for _ in range(self.num_batches):
            
            # 1. 주간 데이터 샘플링
            if len(avai_day_pids) < self.num_day_pids: # 데이터 부족 시 리필 (Oversampling)
                day_dict = self._prepare_batch_indices(self.day_index_dic, self.day_pids)
                avai_day_pids = list(day_dict.keys())
            
            selected_day = random.sample(avai_day_pids, self.num_day_pids)
            for pid in selected_day:
                final_idxs.extend(day_dict[pid].pop(0))
                if not day_dict[pid]: avai_day_pids.remove(pid)

            # 2. 야간 데이터 샘플링
            if len(avai_night_pids) < self.num_night_pids: # 데이터 부족 시 리필
                night_dict = self._prepare_batch_indices(self.night_index_dic, self.night_pids)
                avai_night_pids = list(night_dict.keys())

            selected_night = random.sample(avai_night_pids, self.num_night_pids)
            for pid in selected_night:
                final_idxs.extend(night_dict[pid].pop(0))
                if not night_dict[pid]: avai_night_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

# New add by gu
class RandomIdentitySampler_IdUniform(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            pid = item[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
