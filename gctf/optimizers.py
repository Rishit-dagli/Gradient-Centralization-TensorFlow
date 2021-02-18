import tensorflow as tf

from .centralized_gradients import centralized_gradients_for_optimizer


def adam(learning_rate=0.001,
         beta_1=0.9,
         beta_2=0.999,
         epsilon=1e-7,
         amsgrad=False):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                                         amsgrad=amsgrad)
    optimizer.get_gradients = centralized_gradients_for_optimizer(optimizer)
    return optimizer
