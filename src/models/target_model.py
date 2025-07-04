from abc import ABC, abstractmethod
class TargetModel(ABC):
    @abstractmethod
    def predict(self, data):
        """
        Predict the target variable based on the input data.

        Parameters:
        - data: The input data for prediction.

        Returns:
        - The predicted value.
        """
        pass
    @abstractmethod
    def train(self, data, target):
        """
        Train the model using the provided data and target variable.

        Parameters:
        - data: The input data for training.
        - target: The target variable for training.

        Returns:
        - None
        """
        pass
    
    @abstractmethod
    def evaluate(self, data, target):
        """
        Evaluate the model's performance on the provided data and target variable.

        Parameters:
        - data: The input data for evaluation.
        - target: The target variable for evaluation.

        Returns:
        - A dictionary containing evaluation metrics (e.g., accuracy, precision, recall).
        """
        pass

    @abstractmethod
    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Parameters:
        - filepath: The path where the model should be saved.

        Returns:
        - None
        """
        pass