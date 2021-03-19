from typing import *

import yaml

config_glob: Dict = {}
with open("config/glob/v1.yaml", "r") as f:
    config_glob = yaml.load(f, Loader=yaml.FullLoader)

phases = {"train", "val"}
datasets = {"xview", "xsub"}
