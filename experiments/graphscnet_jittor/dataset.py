import os.path as osp
import random
from typing import Callable

import numpy as np
from typing import Optional

import numpy as np
from numpy import ndarray
import importlib
import pkgutil
def load_ext(name, functions):
    ext_module = importlib.import_module(name)

    for function in functions:
        assert hasattr(ext_module, function), f"Function '{function}' missing in '{name}'."
    return ext_module

ext_module = load_ext("vision3d.ext", ["sample_nodes_with_fps"])


def furthest_point_sample(
    points: ndarray, min_distance: Optional[float] = None, num_samples: Optional[int] = None
) -> ndarray:
    if min_distance is None:
        min_distance = -1.0
    if num_samples is None:
        num_samples = -1
    node_indices = ext_module.sample_nodes_with_fps(points, min_distance, num_samples)
    return node_indices
from array_ops import (
    apply_transform,
    compose_transforms,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_small_transform,
)
from fourdmatch import FourDMatchPairDataset



class TransformFunction(Callable):
    """TransformFunction function.

    1. Read corr data.
    2. Sample nodes with FPS. Compared with RS, FPS tends to push the nodes to the boundary.
    """

    def __init__(self, cfg, subset, use_augmentation):
        subset = subset.split("-")[0]
        self.corr_dir = osp.join(cfg.data.dataset_dir, "correspondences", subset)
        self.use_augmentation = use_augmentation
        self.aug_noise = 0.002
        self.node_coverage = cfg.model.deformation_graph.node_coverage
        self.max_corr=3100

    def __call__(self, data_dict):
        # read corr data
        corr_dict = np.load(osp.join(self.corr_dir, data_dict["filename"]))
        src_corr_points = corr_dict["src_corr_points"].astype(np.float32)
        tgt_corr_points = corr_dict["tgt_corr_points"].astype(np.float32)
        corr_scene_flows = corr_dict["corr_scene_flows"].astype(np.float32)
        corr_labels = corr_dict["corr_labels"].astype(np.int64)

        if self.max_corr is not None and len(src_corr_points)>self.max_corr:
            indices = np.random.permutation(src_corr_points.shape[0])[: self.max_corr]
            src_corr_points = src_corr_points[indices]
            tgt_corr_points = tgt_corr_points[indices]
            corr_scene_flows = corr_scene_flows[indices]
            corr_labels = corr_labels[indices]


        data_dict["src_corr_points"] = src_corr_points.astype(np.float32)
        data_dict["tgt_corr_points"] =tgt_corr_points.astype(np.float32)
        data_dict["corr_scene_flows"] = corr_scene_flows.astype(np.float32)
        data_dict["corr_labels"] = corr_labels.astype(np.int64)


        # augmentation
        if self.use_augmentation:
            src_points = data_dict["src_points"]
            tgt_points = data_dict["tgt_points"]
            scene_flows = data_dict["scene_flows"]
            transform = data_dict["transform"]
            src_corr_points = data_dict["src_corr_points"]
            tgt_corr_points = data_dict["tgt_corr_points"]
            corr_scene_flows = data_dict["corr_scene_flows"]

            deformed_src_points = src_points + scene_flows
            deformed_src_corr_points = src_corr_points + corr_scene_flows
            aug_transform = random_sample_small_transform()
            if random.random() > 0.5:
                tgt_center = tgt_points.mean(axis=0)
                subtract_center = get_transform_from_rotation_translation(
                    None, -tgt_center
                )
                add_center = get_transform_from_rotation_translation(None, tgt_center)
                aug_transform = compose_transforms(
                    subtract_center, aug_transform, add_center
                )
                tgt_points = apply_transform(tgt_points, aug_transform)
                tgt_corr_points = apply_transform(tgt_corr_points, aug_transform)
                transform = compose_transforms(transform, aug_transform)
            else:
                src_center = src_points.mean(axis=0)
                subtract_center = get_transform_from_rotation_translation(
                    None, -src_center
                )
                add_center = get_transform_from_rotation_translation(None, src_center)
                aug_transform = compose_transforms(
                    subtract_center, aug_transform, add_center
                )
                src_points = apply_transform(src_points, aug_transform)
                src_corr_points = apply_transform(src_corr_points, aug_transform)
                deformed_src_points = apply_transform(
                    deformed_src_points, aug_transform
                )
                deformed_src_corr_points = apply_transform(
                    deformed_src_corr_points, aug_transform
                )
                inv_aug_transform = inverse_transform(aug_transform)
                transform = compose_transforms(inv_aug_transform, transform)

            src_points += (
                np.random.rand(src_points.shape[0], 3) - 0.5
            ) * self.aug_noise
            tgt_points += (
                np.random.rand(tgt_points.shape[0], 3) - 0.5
            ) * self.aug_noise
            src_corr_points += (
                np.random.rand(src_corr_points.shape[0], 3) - 0.5
            ) * self.aug_noise
            tgt_corr_points += (
                np.random.rand(tgt_corr_points.shape[0], 3) - 0.5
            ) * self.aug_noise
            scene_flows = deformed_src_points - src_points
            corr_scene_flows = deformed_src_corr_points - src_corr_points

            data_dict["src_points"] = src_points.astype(np.float32)
            data_dict["tgt_points"] = tgt_points.astype(np.float32)
            data_dict["scene_flows"] = scene_flows.astype(np.float32)
            data_dict["src_corr_points"] = src_corr_points.astype(np.float32)
            data_dict["tgt_corr_points"] = tgt_corr_points.astype(np.float32)
            data_dict["corr_scene_flows"] = corr_scene_flows.astype(np.float32)
            data_dict["transform"] = transform.astype(np.float32)

        # sample nodes
        src_points = data_dict["src_points"]
        src_node_indices = furthest_point_sample(
            src_points, min_distance=self.node_coverage
        )
        data_dict["node_indices"] = src_node_indices.astype(np.int64)
        return data_dict


def train_valid_data_loader(cfg):
    train_dataset1 = FourDMatchPairDataset(
        cfg.data.dataset_dir,
        "val",
        start=1,
        transform_fn=TransformFunction(cfg, "val", cfg.train.use_augmentation),
        use_augmentation=False,
        return_corr_indices=cfg.train.return_corr_indices,
    )

    train_loader1=train_dataset1.set_attrs(
            batch_size=1,
            num_workers=cfg.train.num_workers
            )
    train_dataset2 = FourDMatchPairDataset(
        cfg.data.dataset_dir,
        "val",
        start=2,
        transform_fn=TransformFunction(cfg, "val", cfg.train.use_augmentation),
        use_augmentation=False,
        return_corr_indices=cfg.train.return_corr_indices,
    )

    train_loader2=train_dataset2.set_attrs(
            batch_size=1,
            num_workers=cfg.train.num_workers
            )
    valid_dataset = FourDMatchPairDataset(
        cfg.data.dataset_dir,
        "4DMatch",
        transform_fn=TransformFunction(cfg, "4DMatch", False),
        use_augmentation=False,
        return_corr_indices=cfg.test.return_corr_indices,
    )
    valid_loader=valid_dataset.set_attrs(
            batch_size=1,
            num_workers=cfg.train.num_workers
            )
    

    return train_loader1, train_loader2


def test_data_loader(cfg, benchmark):
    test_dataset = FourDMatchPairDataset(
        cfg.data.dataset_dir,
        benchmark,
        start=2,
        transform_fn=TransformFunction(cfg, benchmark, False),
        use_augmentation=False,
        return_corr_indices=cfg.test.return_corr_indices,
        shape_names=cfg.test.shape_names,
    )

    test_loader=test_dataset.set_attrs(
            batch_size=1,
            num_workers=cfg.test.num_workers
            )
    

    return test_loader


def run_test():

    return


if __name__ == "__main__":
    run_test()
