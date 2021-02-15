import tensorflow as tf
import keras.backend as K


def get_centralized_gradients(optimizer, loss, params):
    """
    Compute the centralized gradients.

    Modified version of tf.keras.optimizers.Optimizer.get_gradients
    Reference:
        https://arxiv.org/abs/2004.01461
    """

    # We here just provide a modified get_gradients() function since we are trying to just compute the centralized
    # gradients at this stage which can be used in other optimizers.
    grads = []
    for grad in K.gradients(loss, params):
        grad_len = len(grad.shape)
        if grad_len > 1:
            axis = list(range(grad_len - 1))
            grad -= tf.reduce_mean(grad,
                                   axis=axis,
                                   keep_dims=True)
        grads.append(grad)

    if None in grads:
        raise ValueError('An operation has `None` for gradient. '
                         'Please make sure that all of your ops have a '
                         'gradient defined (i.e. are differentiable). '
                         'Common ops without gradient: '
                         'K.argmax, K.round, K.eval.')
    if hasattr(optimizer, 'clipnorm') and optimizer.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [
            tf.keras.optimizers.clip_norm(
                g,
                optimizer.clipnorm,
                norm) for g in grads]
    if hasattr(optimizer, 'clipvalue') and optimizer.clipvalue > 0:
        grads = [K.clip(g, -optimizer.clipvalue, optimizer.clipvalue)
                 for g in grads]
    return grads


def centralized_gradients_for_optimizer(optimizer):
    """Create a centralized gradients functions for an optimizer.

    # Arguments
        optimizer: a tf.keras.optimizer object. The optimizer you are using.

    # Raises
        TypeError: in case a non optimizer object is passed.
    """

    def get_centralized_gradients_for_optimizer(loss, params):
        return get_centralized_gradients(optimizer, loss, params)

    if 'tensorflow.python.keras.optimizer_v2' in type(optimizer):
        return get_centralized_gradients_for_optimizer
    raise TypeError(str(type(optimizer)) +
                    ' is not a tensorflow.python.keras.optimizer_v2 object')
