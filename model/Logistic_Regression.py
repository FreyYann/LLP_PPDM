import tensorflow as tf
from bin.noise import add_laplace_noise
from bin.processor import disecret_data
from bin.processor import continue_data
import numpy as np
from sklearn.model_selection import train_test_split

from model.Base import LinearModel


class LinearRegression(LinearModel):

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None,noise="add_laplace_noise",n_batches=1000,learning_rate=0.01,
                 batch_size=100):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.noise=noise
        self.n_batches = n_batches
        self.learning_rate = learning_rate
        self.batch_size = batch_size


    def process_data(self,X,y):
        return X, eval(self.noise)(y)



    def optimize(self,X,y):
        X,y=self.process_data(X,y)
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.repeat().shuffle(x_train.shape[0]).batch(self.batch_size)
        optimizer = tf.optimizers.SGD(self.learning_rate)
        for batch_numb, (batch_xs, batch_ys) in enumerate(dataset.take(n_batches), 1):
            gradients = grad(batch_xs, batch_ys)
            optimizer.apply_gradients(zip(gradients, [weights, biases]))

            y_pred = logistic_regression(batch_xs)
            loss = cross_entropy(batch_ys, y_pred)
            acc = accuracy(batch_ys, y_pred)
            print("Batch number: %i, loss: %f, accuracy: %f" % (batch_numb, loss, acc))