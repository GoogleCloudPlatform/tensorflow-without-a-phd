import tensorflow as tf
import numpy as np

import gym
env = gym.make('Pong-v0')

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * 0.9 + r[t]
        discounted_r[t] = running_add
    return discounted_r

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel().reshape((80*80, 1))


# called each time the weights are updated
def generate_single_game_input_fn(env, estimator):
    # generate entire game's worth of (state, label, reward) triples using existing Python code
    def single_game_input_fn():
        state = env.reset()
        
        previous_x = None
        observations = []
        labels = []
        rewards = []

        done = False
        while not done:
            env.render()
            
            current_x = prepro(state)
            observation = current_x - previous_x if previous_x is not None else np.zeros((80*80, 1))
            previous_x = current_x
            
            up_probability = list(estimator.predict(lambda: [observation]))[0]
            
            action = 2 if np.random.uniform() < up_probability else 3
            label = 1.0 if action == 2 else 0.0
            
            state, reward, done, _ = env.step(action)
            
            observations.append(observation)
            labels.append(label)
            rewards.append(reward)
            
        observations = np.array(observations, dtype=np.float64)
        labels = np.array(labels, dtype=np.float64)
        rewards = np.array(rewards, dtype=np.float64)
        final_rewards = discount_rewards(rewards)
            
        return {'observations': observations}, {'labels': labels, 'final_rewards': final_rewards}
    
    return single_game_input_fn


def generate_model_fn():
    def model_fn(features, labels, mode, params=None, config=None):
        observations = tf.identity(features.get('observations'))
        hidden_out = tf.layers.dense(inputs=observations,
                                     units=200,
                                     use_bias=False,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        prob = tf.layers.dense(inputs=hidden_out,
                               units=1,
                               use_bias=False,
                               activation=tf.nn.sigmoid,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        
        
        loss = None
        
        if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN):
            actions = labels.get('labels')
            final_rewards = labels.get('final_rewards')
            sample_cross_entropy = tf.add(
                tf.multiply(actions, tf.log(prob)),
                tf.multiply(tf.subtract(tf.constant(1, dtype=tf.float64), actions), tf.log(tf.subtract(tf.constant(1, dtype=tf.float64), prob))))
            loss = tf.negative(tf.reduce_sum(tf.multiply(final_rewards, sample_cross_entropy)))
        
        
        train_op = None
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(loss)
        
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=prob,
            loss=loss,
            train_op=train_op,
            export_outputs=None
        )
    
    return model_fn


### train

MODEL_DIR = '/tmp/lol'
run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=1)


estimator = tf.estimator.Estimator(generate_model_fn(), model_dir=MODEL_DIR, params=None, config=run_config)
input_fn = generate_single_game_input_fn(env, estimator)

estimator.train(input_fn, steps=3)


