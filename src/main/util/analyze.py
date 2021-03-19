import yaml
from src.main.util.config import config_glob
from termcolor import colored

chosen_classes = config_glob["chosen_class"]
input_data_raw = config_glob["input_data_raw"]

import os


def pp(obj, title=""):
    """
    This for pretty print dictionary.
    """
    print("\n", colored(title, "yellow"))
    for key, value in obj.items():
        print(colored(key, "green"), " : ", sorted(value))


def read_name(filename):
    """
    Extract file name into fields seperately.
    """
    setup_number = int(filename[filename.find("S") + 1 : filename.find("S") + 4])
    camera_id = int(filename[filename.find("C") + 1 : filename.find("C") + 4])
    performer_id = int(filename[filename.find("P") + 1 : filename.find("P") + 4])
    replication_number = int(filename[filename.find("R") + 1 : filename.find("R") + 4])
    action_class = int(filename[filename.find("A") + 1 : filename.find("A") + 4])
    return {
        "setup_number": setup_number,
        "camera_id": camera_id,
        "performer_id": performer_id,
        "replication_number": replication_number,
        "action_class": action_class,
    }


def check_benchmark(
    benchmark=None, filename=None, performer_id=None, setup_number=None, camera_id=None
):
    """
    Each benchmark (xsub or xview) has different samples for training set. This function to check this.
    """
    if filename is not None:
        extracted_name = read_name(filename)

    with open("agcn/config/general-config/general_config.yaml", "r") as f:
        arg = yaml.load(f, Loader=yaml.FullLoader)

    benchmarks = arg["benchmarks"]
    for criteria in benchmarks[benchmark].keys():
        if extracted_name[criteria] not in benchmarks[benchmark][criteria]:
            return False
    return True


def list_info(chosen_classes):
    """
    List out all class, subject, numbody,.... of chosen classes
    """
    # S015C003P025R002A017
    setup_numbers = set()
    camera_ids = set()
    performer_ids = set()
    replication_numbers = set()
    action_classes = set()

    for filename in os.listdir(input_data_raw):
        extracted_name = read_name(filename)
        if extracted_name["action_class"] not in chosen_classes:
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


def check_full_data(chosen_classes):
    """
    Check if each class has full options (full setups, full person, full camera,...)
    """
    for class_ in chosen_classes:
        infos = list_info([class_])
        if (
            infos["setup_number"].__len__ != 32
            or infos["camera_id"].__len__ != 3
            or infos["performer_id"].__len__ != 106
            or infos["replication_number"].__len__ != 2
        ):
            pp(
                infos,
                "[check_full_data] class {} is not full data.".format(
                    infos["action_class"]
                ),
            )


if __name__ == "__main__":
    # show all infos
    infos = list_info(chosen_classes)
    pp(infos, "View of all classes")

    # check full data for chosen classes
    check_full_data(chosen_classes)
