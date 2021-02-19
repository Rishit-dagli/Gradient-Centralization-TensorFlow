import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

from ..optimizers import adam


def get_beta_accumulators(opt, dtype):
    local_step = math_ops.cast(opt.iterations + 1, dtype)
    beta_1_t = math_ops.cast(opt._get_hyper("beta_1"), dtype)
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_t = math_ops.cast(opt._get_hyper("beta_2"), dtype)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    return (beta_1_power, beta_2_power)


def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      lr=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-7):
    lr_t = lr * np.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))

    m_t = beta1 * m + (1 - beta1) * g_t
    v_t = beta2 * v + (1 - beta2) * g_t * g_t

    param_t = param - lr_t * m_t / (np.sqrt(v_t) + epsilon)
    return param_t, m_t, v_t


def testSparse(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with ops.Graph().as_default(), self.cached_session(use_gpu=True):
            # Initialize variables for numpy implementation.
            m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
            var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
            grads0_np = np.array([0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype)
            var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
            grads1_np = np.array([0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype)

            var0 = variables.Variable(var0_np)
            var1 = variables.Variable(var1_np)
            grads0_np_indices = np.array([0, 2], dtype=np.int32)
            grads0 = ops.IndexedSlices(
                constant_op.constant(grads0_np[grads0_np_indices]),
                constant_op.constant(grads0_np_indices), constant_op.constant([3]))
            grads1_np_indices = np.array([0, 2], dtype=np.int32)
            grads1 = ops.IndexedSlices(
                constant_op.constant(grads1_np[grads1_np_indices]),
                constant_op.constant(grads1_np_indices), constant_op.constant([3]))
            opt = adam()
            update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
            self.evaluate(variables.global_variables_initializer())

            # Fetch params to validate initial values
            self.assertAllClose([1.0, 1.0, 2.0], self.evaluate(var0))
            self.assertAllClose([3.0, 3.0, 4.0], self.evaluate(var1))

            beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
            # Run 3 steps of Adam
            for t in range(3):
                self.assertAllCloseAccordingToType(0.9 ** (t + 1),
                                                   self.evaluate(beta_1_power))
                self.assertAllCloseAccordingToType(0.999 ** (t + 1),
                                                   self.evaluate(beta_2_power))
                update.run()

                var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
                var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

                # Validate updated params
                self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
                self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
