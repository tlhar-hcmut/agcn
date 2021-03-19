#Train
class TrainFeederArgs:
    def __init__(self, 
                data_path, 
                label_path,
                debug,
                random_choose,
                random_shift,
                random_move,
                window_size,
                normalization):
        self.data_path=data_path 
        self.label_path=label_path
        self.debug=debug
        self.random_choose=random_choose
        self.random_shift=random_shift
        self.random_move=random_move
        self.window_size=window_size
        self.normalization=normalization

class TestFeederArgs:
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path
        
class GraphArgs:
    def __init__(self, labeling_mode):
        self.labeling_mode=labeling_mode

class ModelArgs:
    def __init__(self, num_class, num_point, num_person, graph, graph_args):
        self.num_class=num_class
        self.num_point=num_point
        self.num_person=num_person
        self.graph=graph
        self.graph_args=graph_args

#Test
class TestFeederArgs:
    def __init__(self, 
                data_path, 
                label_path,
                debug):
        self.data_path=data_path 
        self.label_path=label_path
        self.debug=debug
       