# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple


class IModel:
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance values for given points

        :param X: array of shape (n_points x n_inputs) of points to run prediction for
        :return: Tuple of mean and variance which are 2d arrays of shape (n_points x n_outputs)
        """
        raise NotImplementedError

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model

        :param X: new points
        :param Y: function values at new points X
        """
        raise NotImplementedError

    def optimize(self) -> None:
        """
        Optimize hyper-parameters of model
        """
        raise NotImplementedError

    @property
    def X(self):
        raise NotImplementedError

    @property
    def Y(self):
        raise NotImplementedError


class IDifferentiable:
    def get_prediction_gradients(self, X: np.ndarray) -> Tuple:
        """
        Computes and returns model gradients of mean and variance at given points

        :param X: points to compute gradients at
        """
        raise NotImplementedError


class IPriorHyperparameters:
    def generate_hyperparameters_samples(self, n_samples: np.int) -> np.ndarray:
        """
        Generates the samples from the hyper-parameters of the model.

        :return: numpy array whose rows are samples from the hyper-parameters of the model.
        """
        raise NotImplementedError

    def fix_model_hyperparameters(self,sample_hyperparameters: np.ndarray) -> None:
        """
        Fixes the model hyper-parameters to certain values (which can be taken from samples).

        :param sample_hyperparameters: np.ndarray whose rows contain each hyper-parameters set.
        """
        raise NotImplementedError

    @property
    def hyperparameters_samples(self):
        raise NotImplementedError

