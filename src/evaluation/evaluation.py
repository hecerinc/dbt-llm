'''
This class will be a base class for running steps in the validation pipeline.

The main method that must be implemented is the `run_evaluation` method which takes:
    - input: conversation
    - output: a dict object (easier to serialize), with the evaluation result
'''
from abc import ABC, abstractmethod
from typing import Dict

class Evaluation(ABC):
    name: str = ''
    @abstractmethod
    def run_evaluation(self, conversation: str) -> Dict:
        raise NotImplementedError()

