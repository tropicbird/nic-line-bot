"""### Functions for save and load pickle"""
#pickle functions
import pickle
from chalicelib.param import drive_path
def load_obj(name):
    with open(drive_path + name + '.pkl', 'rb') as f:
        return pickle.load(f)