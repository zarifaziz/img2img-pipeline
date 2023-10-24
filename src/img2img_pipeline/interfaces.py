import abc


class ModelInterface(metaclass=abc.ABCMeta):
    """
    Interface for all models.
    Model classes must implement this interface.
    """

    @abc.abstractmethod
    def predict(self, *args):
        raise NotImplementedError


class PipelineInterface(metaclass=abc.ABCMeta):
    """
    Interface for all pipelines.
    Pipeline classes must implement this interface.
    """

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError
