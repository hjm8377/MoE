import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform, DomainRandomIdentitySampler

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


from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

from .reid import ReID

__factory = {
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

    'reid': ReID,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _, domains = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    domains = torch.tensor(domains, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, domains

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, _ = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DATASETS.NAMES == 'ourapi':
        dataset = OURAPI(root_train=cfg.DATASETS.ROOT_TRAIN_DIR, root_val=cfg.DATASETS.ROOT_VAL_DIR, config=cfg)
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, cfg=cfg)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    print('using img_triplet sampler')
    if cfg.MODEL.DIST_TRAIN:
        print('DIST_TRAIN START')
        mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
        data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=train_collate_fn,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=DomainRandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num

