
"""
2DGS configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig

from twodgs.twodgs import TwodgsModelConfig

"""
Swap out the network config to use OpenCLIP or CLIP here.
"""
twodgs_method = MethodSpecification(
    config=TrainerConfig(
        method_name="2dgs",
        steps_per_eval_batch=0,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(#use this for overlaying dino on top of a garfield trained model
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            ),
            model=TwodgsModelConfig(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "dino_feats": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=30000,
                ),
            },
            "nn_projection": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=30000,
                ),
            },
            "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-6, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-8, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
            "bilateral_grid": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for 2dgs",
)
