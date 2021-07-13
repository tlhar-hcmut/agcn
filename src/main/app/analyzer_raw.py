import os

from src.main.config import cfg_ds
from src.main.generator import util
from src.main.util import pprinter


def list_info(chosen_classes, path_sample_ignore):
    '''
    List out all class, subject, numbody,.... of chosen classes
    '''
    # S015C003P025R002A017
    setup_numbers = set()
    camera_ids = set()
    performer_ids = set()
    replication_numbers = set()
    action_classes = set()
    num_samples=0

    ls_sample_ignore=[]
    if path_sample_ignore!=None:
        with open(path_sample_ignore, 'r') as f:
            ls_sample_ignore =[line.strip()+".skeleton" for line in f.readlines()]

    for filename in os.listdir(input_data_raw):
        if filename in ls_sample_ignore:
            continue
        extracted_name = util.read_meta_data(filename)
        if (extracted_name['action_class'] not in chosen_classes):
            continue

        setup_numbers.add(extracted_name['setup_number'])
        camera_ids.add(extracted_name['camera_id'])
        performer_ids.add(extracted_name['performer_id'])
        replication_numbers.add(extracted_name['replication_number'])
        action_classes.add(extracted_name['action_class'])
        num_samples+=1

    return {
        'setup_number': setup_numbers,
        'camera_id': camera_ids,
        'performer_id': performer_ids,
        'replication_number': replication_numbers,
        'action_class': action_classes,
        'num_sample': num_samples
    }


def check_full_data(chosen_classes, ls_sample_ignore):
    '''
    Check if each class has full options (full setups, full person, full camera,...)
    '''
    for class_ in chosen_classes:
        infos = list_info([class_], ls_sample_ignore)
        if infos['setup_number'].__len__ != 32\
                or infos['camera_id'].__len__ != 3\
                or infos['performer_id'].__len__ != 106\
                or infos['replication_number'].__len__ != 2:
            pprinter.pp(infos, '[check_full_data] class {} is not full data.'.format(infos['action_class']))


if __name__ == "__main__":

    chosen_classes = cfg_ds.ls_class
    input_data_raw = cfg_ds.path_data_raw
    sample_ignore=cfg_ds.path_data_ignore

    infos = list_info(chosen_classes, sample_ignore)
    pprinter.pp(infos, "View of all classes")

    check_full_data(chosen_classes, sample_ignore)
