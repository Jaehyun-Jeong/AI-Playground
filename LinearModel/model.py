import numpy as np


class LinearModel:

    def __init__(
        self,
        is_train: bool = True,
        learning_rate: float = 0.0001
    ):
        # Init all weight to 0
        self.W = np.array([1, 0.1, 0.3])

        # Gradients
        self.__model_grad = 0
        self.__sigmoid_grad = 0
        self.__binary_cross_entropy_grad = 0

        # Train parameters
        self.__is_train = is_train
        self.__learning_rate = learning_rate

    @property
    def is_train(self):
        return self.__is_train

    @is_train.setter
    def is_train(
        self,
        is_train: bool
    ):
        self.__is_train = is_train

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(
        self,
        learning_rate: float
    ):
        self.__learning_rate = learning_rate

    def model(
        self,
        data: np.ndarray
    ):
        # Get the number of data
        data_size = data.shape[0]
        ones = np.ones(data_size)  # For bias multiplication

        # Add column of ones to multiply bias
        data = np.column_stack((ones, data))

        # Save grads
        if self.__is_train:
            # (|N|, features)
            self.__model_grad = data

        # (|N|, )
        return np.matmul(data, self.W)

    def sigmoid(
        self,
        arr: np.ndarray
    ):
        # Save grads
        if self.__is_train:
            # (|N|, )
            self.__sigmoid_grad = np.exp(-arr) / (1 + np.exp(-arr))**2

        # (|N|, )
        return 1 / (1 + np.exp(-arr))

    def binary_cross_entropy(
        self,
        Y_hat: np.ndarray,
        Y: np.ndarrady
    ):
        if self.__is_train:
            # (|N|, )
            self.__binary_cross_entropy_grad = \
                (Y_hat - Y) / (Y_hat*(1 - Y_hat))

        # (|N|, )
        return -{Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat)}

    def batch_gradient_descent(
        self
    ):
        temp_grad = np.ndarray([0, 0, 0])
        grad = np.ndarray([0, 0, 0])
        batch_size = self.__model_grad.shape[0]

        for idx, model_grad in enumerate(self.__model_grad):
            temp_grad = np.copy(model_grad)
            temp_grad = self.__sigmoid_grad[idx] * temp_grad
            temp_grad = self.__binary_cross_entropy_grad[idx] * temp_grad
            grad = grad + temp_grad

        # Get average of the gradients
        grad = grad / batch_size

        return grad

    def update_weight(
        self,
        data: np.ndarray
    ):
        pass


if __name__ == "__main__":
    # data = np.array([[1, 2], [3, 2], [10, 2.3]])
    data = np.array([[1, 0], [3, 2], [10, 2.3]])

    linear_model = LinearModel()

    print(linear_model.sigmoid(data))
