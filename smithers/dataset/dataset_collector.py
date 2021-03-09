import os, sys
#from smithers.dataset.abstract_dataset import AbstractDataset
import smithers.dataset.abstract_dataset


class DatasetCollector(object):

    def search(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

        for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
            mod = __import__('smithers.dataset', fromlist=['datasets'])
            classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]

            classes = [
                    cls for cls in classes 
                    if cls.__base__ == smithers.dataset.abstract_dataset.AbstractDataset]

        return classes

    def __init__(self):
        pass
