import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfm = tf.math
tfd = tfp.distributions
tfb = tfp.bijectors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Distribution to model:
# p(x1,x2) = N(x1|mu=(1/4)*(x2**2),sigma=1)*N(x2|mu=0,sigma=4)
batch_size = 512 * 2

n2 = tfd.Normal(loc=0., scale=4.)
x2_samples = n2.sample(batch_size)
n1 = tfd.Normal(loc=0.25 * (x2_samples ** 2), scale=1)
x1_samples = n1.sample()
x_samples = tf.stack([x1_samples, x2_samples], axis=1)

# Verify distribution
plt.figure()
plt.hexbin(x_samples[:, 0], x_samples[:, 1], C=n1.prob(x1_samples) * n2.prob(x2_samples), cmap='rainbow')

## =========
# Inference

# Base dist, isotropic gaussian
q0 = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])


# PReLU as flow
# PReLU(x) = x if x > 0, else alpha * x
class PReLU(tfb.Bijector):
    def __init__(self, alpha=.5):
        super(PReLU, self).__init__(event_ndims=1)
        self.alpha = alpha

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, 1. / (self.alpha * y))

    def inverse_log_det_jacobian(self, y):
        event_dims = self._event_dims_tensor(y)
        I = tf.ones_like(y)
        J_inv = tf.where(tf.greater_equal(y, 0), I, 1.0 / self.alpha * I)
        # abs is actually redundant here, since this det Jacobian is > 0
        log_abs_det_J_inv = tf.math.log(tf.abs(J_inv))
        return tf.reduce_sum(log_abs_det_J_inv, axis=event_dims)


class Flows(tf.keras.models.Model):

    def __init__(self):
        super().__init__()
        self.bijectors = list()
        self.shift = tf.Variable([0.], dtype=tf.float32, name='shift')
        self.scale = tf.Variable([10.], dtype=tf.float32, name='scale')
        self.bijector = tfb.Exp()(tfb.AffineScalar(shift=self.shift, scale=self.scale))
        self.model = tfd.TransformedDistribution(
            distribution=tfd.Uniform(),
            bijector=self.bijector,
            event_shape=(2,))


class LossModel(tf.keras.models.Model):

    def __init__(self, bijector_layer):
        super().__init__()
        self.model = bijector_layer

    def call(self, *x):
        return dict(loss=tf.reduce_mean(self.model.model.log_prob(x)))


class TrainStepper():
    def __init__(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        self._debug = False
        self._tfcall = tf.function(self._call)
        self.step = 0

    def debug(self, value=None):
        if value is None:
            self._debug = not self._debug
        else:
            self._debug = value
        print(f'debug={self.debug}')

    def __call__(self, *data):
        self.step += 1
        if self._debug:
            return self._call(*data)
        else:
            return self._tfcall(*data)

    def _call(self, *data):
        with tf.GradientTape() as tape:
            d = self.model(*data)
        print(self.model.trainable_variables)
        gradients = tape.gradient(d['loss'], self.model.trainable_variables)
        _ = self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return d

    def train(self, data, epochs=1000, log_period=100):
        for epoch in range(epochs):
            d = self(*data)
            if self.step % log_period == 0:
                print({k: v.numpy() for k, v in d.items()})
                for k, v in d.items():
                    tf.summary.scalar(k, v, step=self.step)





x = np.random.rand(100, 1).astype('float32') * 2.34 + 7.3

# TODO
# update MyLayer and LossModel with affine bijector + change x + change loss -> should be -tf.reduce_mean(dist.log_prob(x_samples)) where dist is the resulting distribution
# see also https://stackoverflow.com/questions/57261612/better-way-of-building-realnvp-layer-in-tensorflow-2-0 for reference
mylayer = MyLayer()
lossmodel = LossModel(mylayer)
optimizer = tf.optimizers.Adam(learning_rate=0.001)
stepper = TrainStepper(optimizer=optimizer, model=lossmodel)

stepper.train(x)

final_samples = mylayer.model.sample(batch_size)
plt.figure()
plt.hexbin(final_samples[:, 0], final_samples[:, 1], C=mylayer.model.prob(final_samples), cmap='rainbow')

plt.show()
