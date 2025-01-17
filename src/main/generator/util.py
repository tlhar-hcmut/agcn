import os
from typing import Dict, List

from src.main.config.structure import BenchmarkConfig

class ClassOrder:
    new_order_map = {}

    @staticmethod
    def reorder(list_class:list):
        max_idx=0

        for class_ in list_class:
            if class_ not in ClassOrder.new_order_map:
                ClassOrder.new_order_map[class_]=max_idx
                max_idx+=1
        return ClassOrder.new_order_map
def read_meta_data(filename: str) -> Dict:
    """
    Extract file name into fields seperately.
    """
    setup_number = int(filename[filename.find("S") + 1: filename.find("S") + 4])
    camera_id = int(filename[filename.find("C") + 1: filename.find("C") + 4])
    performer_id = int(filename[filename.find("P") + 1: filename.find("P") + 4])
    replication_number = int(filename[filename.find("R") + 1: filename.find("R") + 4])
    action_class = int(filename[filename.find("A") + 1: filename.find("A") + 4])
    return {
        "setup_number": setup_number,
        "camera_id": camera_id,
        "performer_id": performer_id,
        "replication_number": replication_number,
        "action_class": action_class,
    }


def list_info(path_data_raw: str, ls_class: List[int] = None) -> Dict:
    """
    List out all class, subject, numbody,.... of chosen classes
    """
    setup_numbers = set()
    camera_ids = set()
    performer_ids = set()
    replication_numbers = set()
    action_classes = set()

    for filename in os.listdir(path_data_raw):
        extracted_name = read_meta_data(filename)
        if ls_class != None and extracted_name["action_class"] not in ls_class:
            continue

        setup_numbers.add(extracted_name["setup_number"])
        camera_ids.add(extracted_name["camera_id"])
        performer_ids.add(extracted_name["performer_id"])
        replication_numbers.add(extracted_name["replication_number"])
        action_classes.add(extracted_name["action_class"])

    return {
        "setup_number": setup_numbers,
        "camera_id": camera_ids,
        "performer_id": performer_ids,
        "replication_number": replication_numbers,
        "action_class": action_classes,
    }


def check_benchmark(filename: str, benchmark: BenchmarkConfig) -> bool:
    """
    Each benchmark (xsub or xview) has different samples for training set. This function to check this.
    """
    dict_meta_data = read_meta_data(filename)
    for field in dict_meta_data.keys():
        if (
            len(benchmark.__dict__[field]) != 0 and
            dict_meta_data[field] not in benchmark.__dict__[field]
        ):
            return False

    return True
