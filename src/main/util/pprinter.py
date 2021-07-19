from termcolor import colored
from collections.abc import Iterable


def pp(obj=None, title=""):
    '''
    This for pretty print dictionary.
    '''
    print("\n",colored(title,"yellow"),"\n")
    if obj==None:
        return
    for key, value in obj.items():
        if isinstance(value, Iterable):
            print(colored(key, 'green'), ' : ', sorted(value))
        else:
            print(colored(key, 'green'), ' : ', value)


def pp_scalar(obj=None, title=""):
    '''
    This for pretty print scalar.
    '''
    print(colored(title, "yellow"), "\n")
    if obj == None:
        return
    print(colored(obj, 'green'))
    print("\n")
