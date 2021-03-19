from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkConfig:
    name: str
    setup_number: List[int]
    camera_id: List[int]
    performer_id: List[int]
    replication_number: List[int]
    action_class: List[int]


@dataclass
class DatasetConfig:
    path_data_raw: str
    path_data_preprocess: str
    path_data_ignore: str
    path_visualization: str
    ls_benmark: List[BenchmarkConfig]
    ls_class: List[int]
    num_body: int
    num_joint: int
    num_frame: int
    max_body: int
