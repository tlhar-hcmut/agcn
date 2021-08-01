from dataclasses import dataclass, field
from typing import List, Any, Dict


@dataclass
class BenchmarkConfig:
    name: str
    setup_number: List[int]
    camera_id: List[int]
    performer_id: List[int]
    replication_number: List[int]
    action_class: List[int]


@dataclass
class TKHARConfig:
    name            :str
    desc            :str
    
    benchmark       :str

    path_data_raw: str
    path_data_preprocess: str
    path_data_ignore: str
    path_visualization: str
    ls_benmark: List[BenchmarkConfig]
    ls_class =[3, 4, 7, 8, 9, 10, 21, 23, 27, 28, 93, 102]
    num_body =2
    num_joint:int
    num_frame=300
    max_body=4

    output_train    :str
    input_size      :tuple
    len_feature_new :list
    num_block       :int
    dropout         :float
    num_head        :int
    loss            ="crossentropy"    
    optim           ="adam"
    stream          :list = None
    input_size_temporal: tuple =None
    optim_cfg       :Dict[str, object] = field(default_factory=lambda: {}) #to avoid use the same dictionary (immutable) for all objects
    batch_size      :int = 8
    pretrained_path :str = None
    num_of_epoch    :int = 120
    num_class       :int = 12
