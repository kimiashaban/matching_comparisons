from .base import Metric, ReferenceFreeMetric, ReferenceBasedMetric
from .clip_score import CLIPScoreMetric
from .aesthetic import AestheticScoreMetric
from .image_reward_metric import ImageRewardMetric
from .hps import HPSv21Metric, HPSv2Metric
from .fid import FIDMetric
from .patch_fid import PatchFIDMetric
from .kid import KIDMetric
from .pick_score import PickScoreMetric
from .arniqa_metric import ARNIQAMetric
from .clipiqa_metric import CLIPIQAMetric
from .liqe_metric import LIQEMetric
from .musiq_metric import MUSIQMetric
from .niqe_metric import NIQEMetric
from .nrqm_metric import NRQMMetric
from .topiq_metric import TOPIQMetric

METRIC_REGISTRY: dict[str, type[Metric]] = {
    "clip_score": CLIPScoreMetric,
    "aesthetic": AestheticScoreMetric,
    "image_reward": ImageRewardMetric,
    "hps_v2": HPSv2Metric,
    "hps_v2_1": HPSv21Metric,
    "fid": FIDMetric,
    "patch_fid": PatchFIDMetric,
    "kid": KIDMetric,
    "pick_score": PickScoreMetric,
    "arniqa": ARNIQAMetric,
    "clipiqa": CLIPIQAMetric,
    "liqe": LIQEMetric,
    "musiq": MUSIQMetric,
    "niqe": NIQEMetric,
    "nrqm": NRQMMetric,
    "topiq_nr": TOPIQMetric,
}

REFERENCE_FREE_METRICS = {k for k, v in METRIC_REGISTRY.items() if issubclass(v, ReferenceFreeMetric)}
REFERENCE_BASED_METRICS = {k for k, v in METRIC_REGISTRY.items() if issubclass(v, ReferenceBasedMetric)}

__all__ = [
    "Metric", "ReferenceFreeMetric", "ReferenceBasedMetric",
    "METRIC_REGISTRY", "REFERENCE_FREE_METRICS", "REFERENCE_BASED_METRICS",
]
