"""
Module for abstract class `Dataset`
"""

from abc import ABC, abstractmethod
 
class AbstractDataset(ABC):
 
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def time_dependent(self):
        pass

    @property
    @abstractmethod
    def parametric(self):
        pass

    @property
    @abstractmethod
    def description(self):
        pass
