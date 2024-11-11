# can be placed in train file itself
# this seems too correct and may not look practically implemented, we can change it to only have below commented one instead,
# but it would definetly affect performance

# def cosine_schedule(initial_lr, min_lr, total_steps, first_decay_steps, t_mul=2.0, m_mul=1.0):
#     alpha = min_lr / initial_lr  
    
#     cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
#         initial_learning_rate=initial_lr,
#         first_decay_steps=first_decay_steps,
#         t_mul=t_mul,
#         m_mul=m_mul,
#         alpha=alpha
#     )
#     return cosine_decay

   


import tensorflow as tf


def cosine_schedule(initial_lr, min_lr, total_steps, first_decay_steps, t_mul=2.0, m_mul=1.0):
    alpha = min_lr / initial_lr
    
    # Calculate the number of expected restarts
    n_restarts = tf.math.floor(tf.math.log(1 - (total_steps / first_decay_steps) * (1 - t_mul)) / tf.math.log(t_mul))
    
    # Adjust first_decay_steps to fit total_steps
    adjusted_first_decay_steps = total_steps / ((1 - t_mul ** (n_restarts + 1)) / (1 - t_mul))

    cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=int(adjusted_first_decay_steps),
        t_mul=t_mul,
        m_mul=m_mul,
        alpha=alpha
    )

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            if step < total_steps:
                return cosine_decay(step)
            else:
                return min_lr

    return CustomSchedule()
