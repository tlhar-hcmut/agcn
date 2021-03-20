from termcolor import colored

def pp(obj=None, title=""):
    '''
    This for pretty print dictionary.
    '''
    output = print("\n",colored(title,"yellow"),"\n")
    if obj==None:
        return
    for key, value in obj.items():
        print(colored(key, 'green'), ' : ', sorted(value))