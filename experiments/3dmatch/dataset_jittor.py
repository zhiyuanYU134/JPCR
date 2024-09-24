from random import shuffle
from PCR_Jittor.jittor.datasets.registration.threedmatch.dataset import JittorThreeDMatchPairDataset,Neighbor_limits
from PCR_Jittor.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def train_valid_data_loader(cfg):
    Neighbor_dataset = Neighbor_limits(
        cfg,
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        Neighbor_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    print(neighbor_limits)
    del Neighbor_dataset
    train_dataset0 = JittorThreeDMatchPairDataset(
        cfg,
        True,
        cfg.data.dataset_root,
        'train',
        neighbor_limits=neighbor_limits,
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    train_loader0=train_dataset0.set_attrs(
            batch_size=1,
            num_workers=cfg.train.num_workers
            )
    
    train_dataset1 = JittorThreeDMatchPairDataset(
        cfg,
        False,
        cfg.data.dataset_root,
        'train',
        neighbor_limits=neighbor_limits,
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    train_loader1=train_dataset1.set_attrs(
            batch_size=1,
            num_workers=cfg.train.num_workers
            )

    return train_loader0, train_loader1

""" ,
            shuffle=True """

def test_data_loader(cfg,benchmark):
    Neighbor_dataset = Neighbor_limits(
        cfg,
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        Neighbor_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    print(neighbor_limits)
    del Neighbor_dataset
    train_dataset0 = JittorThreeDMatchPairDataset(
        cfg,
        True,
        cfg.data.dataset_root,
        benchmark,
        neighbor_limits=neighbor_limits
    )
    train_loader0=train_dataset0.set_attrs(
            batch_size=1,
            num_workers=cfg.test.num_workers
            )
    
  

    return train_loader0