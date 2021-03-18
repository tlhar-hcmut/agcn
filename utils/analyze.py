import yaml
with open("agcn/config/general-config/general_config.yaml", 'r') as f:
    arg = yaml.load(f, Loader=yaml.FullLoader)
chosen_classes = arg['chosen_class']
input_data_raw =arg['input_data_raw']

import os
from  agcn.utils.utils  import read_name
import pprint

def list_info(chosen_classes):
    '''
    List out all class, subject, numbody,.... of chosen classes
    '''
    # S015C003P025R002A017
    setup_numbers = set()
    camera_ids = set() 
    performer_ids = set()
    replication_numbers = set()
    action_classes=set()

    for filename in os.listdir(input_data_raw):
        extracted_name = read_name(filename)
        if (extracted_name['action_class'] not in chosen_classes): 
            continue 

        setup_numbers.add(extracted_name['setup_number'])
        camera_ids.add(extracted_name['camera_id'])
        performer_ids.add(extracted_name['performer_id'])
        replication_numbers.add(extracted_name['replication_number'])
        action_classes.add(extracted_name['action_class'])
    
    return {
            'setup_number':setup_numbers,
            'camera_id':camera_ids,
            'performer_id': performer_ids,
            'replication_number':replication_numbers,
            'action_class':action_classes
            }
   

def check_full_data(chosen_classes):
    '''
    Check if each class has full options (full setups, full person, full camera,...)
    '''
    pp = pprint.PrettyPrinter(indent=4, width=400)
    for class_ in chosen_classes:
        infos=list_info([class_])
        if  infos['setup_number'].__len__!= 32\
            or infos['camera_id'].__len__!= 3\
            or infos['performer_id'].__len__!= 106\
            or infos['replication_number'].__len__!= 2:
            print('\n[check_full_data] class {} is not full data.'.format(infos['action_class']))
            pp.pprint(infos)
                


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4, width=500)

    #show all infos
    infos = list_info(chosen_classes)
    pp.pprint(infos)

    #check full data for chosen classes
    check_full_data(chosen_classes)

