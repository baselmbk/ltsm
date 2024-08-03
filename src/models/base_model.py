from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def build(self):
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions using the trained model."""
        pass

    @abstractmethod
    def save_model(self, filepath):
        """Save the model to a file."""
        pass

    @abstractmethod
    def load_model(self, filepath):
        """Load the model from a file."""
        pass