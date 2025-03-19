class NotFittedError(Exception):
    """Exception raised when an estimator is used before fitting."""

    def __init__(self, message: str = "Model not fitted. Call 'fit' first.") -> None:
        """Initialize of NotFittedError.

        Args:
            message (str, optional): error message.
                Defaults to "Model not fitted. Call 'fit' first.".

        """
        self.message = message
        super().__init__(self.message)

class ValidationError(Exception):
    """Exception raised when input data validation fails."""

    def __init__(self, message: str = "Input data validation failed.") -> None:
        """Initialize of ValidationError.

        Args:
            message (str, optional): Error message.
                Defaults to "Input data validation failed.".

        """
        self.message = message
        super().__init__(self.message)


class ParameterError(Exception):
    """Exception raised when invalid hyperparameters are provided."""

    def __init__(self, message: str = "Invalid hyperparameters provided.") -> None:
        """Initialize of ParameterError.

        Args:
            message (str, optional): Error message.
                Defaults to "Invalid hyperparameters provided.".

        """
        self.message = message
        super().__init__(self.message)
