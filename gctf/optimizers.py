import tensorflow as tf

from .centralized_gradients import centralized_gradients_for_optimizer


def update_optimizer(optimizer):
    optimizer.get_gradients = centralized_gradients_for_optimizer(optimizer)
    return optimizer


def adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07):
    optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=initial_accumulator_value,
        epsilon=epsilon)
    return update_optimizer(optimizer)


def adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07):
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate,
                                             rho=rho,
                                             epsilon=epsilon)
    return update_optimizer(optimizer)


def adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        amsgrad=amsgrad)
    return update_optimizer(optimizer)


def adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate,
                                           beta_1=beta_1,
                                           beta_2=beta_2,
                                           epsilon=epsilon)
    return update_optimizer(optimizer)


def ftrl(
        learning_rate=0.001,
        learning_rate_power=-0.5,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0,
        l2_shrinkage_regularization_strength=0.0,
        beta=0.0):
    optimizer = tf.keras.optimizers.Adamax(
        learning_rate=learning_rate,
        learning_rate_power=learning_rate_power,
        initial_accumulator_value=initial_accumulator_value,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength,
        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
        beta=beta)
    return update_optimizer(optimizer)


def nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon)
    return update_optimizer(optimizer)


def rmsprop(
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-07,
        centered=False):
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon,
        centered=centered)
    return update_optimizer(optimizer)


def sgd(learning_rate=0.01, momentum=0.0, nesterov=False):
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov)
    return update_optimizer(optimizer)
