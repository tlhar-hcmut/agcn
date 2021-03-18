import yaml 
from termcolor import colored

def read_name(filename):
    '''
    Extract file name into fields seperately.
    '''
    setup_number    = int(filename[filename.find('S') + 1:filename.find('S') + 4])
    camera_id       = int(filename[filename.find('C') + 1:filename.find('C') + 4])
    performer_id    = int(filename[filename.find('P') + 1:filename.find('P') + 4])
    replication_number = int(filename[filename.find('R') + 1:filename.find('R') + 4])
    action_class    = int(filename[filename.find('A') + 1:filename.find('A') + 4])
    return {
            'setup_number':setup_number,
            'camera_id':camera_id,
            'performer_id': performer_id,
            'replication_number':replication_number,
            'action_class':action_class
            }

def checkBenchmark(benchmark=None, filename=None, performer_id=None, setup_number=None, camera_id=None):
    '''
    Each benchmark (xsub or xview) has different samples for training set. This function to check this.
    '''
    if filename is not None:
        extracted_name  = read_name(filename)
       
    with open("agcn/config/general-config/general_config.yaml", 'r') as f:
        arg = yaml.load(f, Loader=yaml.FullLoader)
    
    benchmarks = arg['benchmarks']
    for criteria in benchmarks[benchmark].keys():
         if extracted_name[criteria] not in benchmarks[benchmark][criteria]:
             return False
    return True



def pp(obj, title=""):
    '''
    This for pretty print dictionary.
    '''
    print("\n",colored(title,"yellow"))
    for key, value in obj.items():
        print(colored(key, 'green'), ' : ', sorted(value))