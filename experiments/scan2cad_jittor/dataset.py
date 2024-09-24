from PCR_Jittor.jittor.datasets.registration.scan2cad.dataset import Scan2cadKPConvDataset,Neighbor_limits
from PCR_Jittor.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def train_valid_data_loader(cfg):
    config=cfg
    Neighbor_dataset = Neighbor_limits(
        cfg,
        config.data.jittor_scan2cad_root,
        split='train',
        matching_radius=cfg.model.ground_truth_matching_radius,
        max_point=config.train.point_limit,
        use_augmentation=config.train.use_augmentation,
        augmentation_noise=config.train.augmentation_noise,
        rotation_factor=config.train.augmentation_rotation
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
    train_dataset0 = Scan2cadKPConvDataset(
        cfg,
        config.data.jittor_scan2cad_root,
        split='train',
        matching_radius=cfg.model.ground_truth_matching_radius,
        neighbor_limits=neighbor_limits,
        max_point=config.train.point_limit,
        use_augmentation=config.train.use_augmentation,
        augmentation_noise=config.train.augmentation_noise,
        rotation_factor=config.train.augmentation_rotation
    )
    train_loader0=train_dataset0.set_attrs(
            batch_size=1,
            num_workers=cfg.train.num_workers
            )

    return train_loader0


def test_data_loader(cfg):
    config=cfg
    Neighbor_dataset = Neighbor_limits(
        cfg,
        config.data.jittor_scan2cad_root,
        split='train',
        matching_radius=cfg.model.ground_truth_matching_radius,
        max_point=config.train.point_limit,
        use_augmentation=config.train.use_augmentation,
        augmentation_noise=config.train.augmentation_noise,
        rotation_factor=config.train.augmentation_rotation
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
    train_dataset0 = Scan2cadKPConvDataset(
        cfg,
        config.data.jittor_scan2cad_root,
        split='test',
        matching_radius=cfg.model.ground_truth_matching_radius,
        neighbor_limits=neighbor_limits,
        max_point=config.test.point_limit
    )
    train_loader0=train_dataset0.set_attrs(
            batch_size=1,
            num_workers=cfg.test.num_workers
            )
    
    return train_loader0

