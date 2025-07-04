class ModelRegistry:
    """
    A registry for managing machine learning models, including their training, evaluation, and deployment.
    """

    def __init__(self):
        self.models = {}

    def register_model(self, model_name, model):
        """
        Register a new model in the registry.

        Args:
            model_name (str): The name of the model.
            model: The trained model object.
        """
        if model_name in self.models:
            raise ValueError(f"Model '{model_name}' is already registered.")
        self.models[model_name] = model

    def get_model(self, model_name):
        """
        Retrieve a registered model from the registry.

        Args:
            model_name (str): The name of the model to retrieve.

        Returns:
            The registered model object.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in the registry.")
        return self.models[model_name]

    def list_models(self):
        """
        List all registered models in the registry.

        Returns:
            list: A list of names of all registered models.
        """
        return list(self.models.keys())

    def unregister_model(self, model_name):
        """
        Unregister a model from the registry.

        Args:
            model_name (str): The name of the model to unregister.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in the registry.")
        del self.models[model_name]