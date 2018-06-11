# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import tensorflow as tf
import numpy as np
import gym
from builtins import input

from trainer.helpers import discount_rewards, prepro
from agents.tools.wrappers import AutoReset, FrameHistory
from collections import deque

# Open AI gym Atari env: 0: 'NOOP', 2: 'UP', 3: 'DOWN'
ACTIONS = [0, 2, 3]

OBSERVATION_DIM = 80 * 80

MEMORY_CAPACITY = 100000
ROLLOUT_SIZE = 10000


# MEMORY stores tuples:
# (observation, label, reward)
MEMORY = deque([], maxlen=MEMORY_CAPACITY)
def gen():
    for m in list(MEMORY):
        yield m


def build_graph(observations):
    """Calculates logits from the input observations tensor.
    This function will be called twice: rollout and train.
    The weights will be shared.
    """
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        hidden = tf.keras.layers.Dense(args.hidden_dim, use_bias=False, activation='relu')(observations)
        logits = tf.keras.layers.Dense(len(ACTIONS), use_bias=False)(hidden)

    return logits


def main(args):
    args_dict = vars(args)
    print('args: {}'.format(args_dict))

    with tf.Graph().as_default() as g:
        # rollout subgraph
        with tf.name_scope('rollout'):
            observations = tf.placeholder(shape=(None, OBSERVATION_DIM), dtype=tf.float32)
            
            logits = build_graph(observations)

            logits_for_sampling = tf.reshape(logits, shape=(1, len(ACTIONS)))

            # Sample the action to be played during rollout.
            sample_action = tf.squeeze(tf.multinomial(logits=logits_for_sampling, num_samples=1))
        
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=args.learning_rate,
            decay=args.decay
        )

        # dataset subgraph for experience replay
        with tf.name_scope('dataset'):
            # the dataset reads from MEMORY
            ds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32, tf.float32))
            ds = ds.shuffle(MEMORY_CAPACITY).repeat().batch(args.batch_size)
            iterator = ds.make_one_shot_iterator()

        # training subgraph
        with tf.name_scope('train'):
            # the train_op includes getting a batch of data from the dataset, so we do not need to use a feed_dict when running the train_op.
            next_batch = iterator.get_next()
            train_observations, labels, processed_rewards = next_batch

            # This reuses the same weights in the rollout phase.
            train_observations.set_shape((args.batch_size, OBSERVATION_DIM))
            train_logits = build_graph(train_observations)

            cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=train_logits,
                labels=labels
            )

            # Extra loss when the paddle is moved, to encourage more natural moves.
            probs = tf.nn.softmax(logits=train_logits)
            move_cost = args.laziness * tf.reduce_sum(probs * [0, 1.0, 1.0], axis=1)

            loss = tf.reduce_sum(processed_rewards * cross_entropies + move_cost)

            global_step = tf.train.get_or_create_global_step()

            train_op = optimizer.minimize(loss, global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=args.max_to_keep)

        with tf.name_scope('summaries'):
            rollout_reward = tf.placeholder(
                shape=(),
                dtype=tf.float32
            )

            # the weights to the hidden layer can be visualized
            hidden_weights = tf.trainable_variables()[0]
            for h in range(args.hidden_dim):
                slice_ = tf.slice(hidden_weights, [0, h], [-1, 1])
                image = tf.reshape(slice_, [1, 80, 80, 1])
                tf.summary.image('hidden_{:04d}'.format(h), image)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
                tf.summary.scalar('{}_max'.format(var.op.name), tf.reduce_max(var))
                tf.summary.scalar('{}_min'.format(var.op.name), tf.reduce_min(var))
                
            tf.summary.scalar('rollout_reward', rollout_reward)
            tf.summary.scalar('loss', loss)

            merged = tf.summary.merge_all()

    inner_env = gym.make('Pong-v0')
    # tf.agents helper to more easily track consecutive pairs of frames
    env = FrameHistory(inner_env, past_indices=[0, 1], flatten=False)
    # tf.agents helper to automatically reset the environment
    env = AutoReset(env)

    with tf.Session(graph=g) as sess:
        if args.restore:
            restore_path = tf.train.latest_checkpoint(args.output_dir)
            print('Restoring from {}'.format(restore_path))
            saver.restore(sess, restore_path)
        else:
            sess.run(init)

        summary_path = os.path.join(args.output_dir, 'summary')
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

        # lowest possible score after an episode as the
        # starting value of the running reward
        _rollout_reward = -21.0

        for i in range(args.n_epoch):
            print('>>>>>>> epoch {}'.format(i+1))

            print('>>> Rollout phase')
            epoch_memory = []
            episode_memory = []

            # The loop for actions/stepss
            _observation = np.zeros(OBSERVATION_DIM)
            while True:
                # sample one action with the given probability distribution
                _label = sess.run(sample_action, feed_dict={observations: [_observation]})

                _action = ACTIONS[_label]

                _pair_state, _reward, _done, _ = env.step(_action)

                if args.render:
                    env.render()
                
                # record experience
                episode_memory.append((_observation, _label, _reward))

                # Get processed frame delta for the next step
                pair_state = _pair_state

                current_state, previous_state = pair_state
                current_x = prepro(current_state)
                previous_x = prepro(previous_state)

                _observation = current_x - previous_x

                if _done:
                    obs, lbl, rwd = zip(*episode_memory)

                    # processed rewards
                    prwd = discount_rewards(rwd, args.gamma)
                    prwd -= np.mean(prwd)
                    prwd /= np.std(prwd)

                    # store the processed experience to memory
                    epoch_memory.extend(zip(obs, lbl, prwd))
                    
                    # calculate the running rollout reward
                    _rollout_reward = 0.9 * _rollout_reward + 0.1 * sum(rwd)

                    episode_memory = []

                    if args.render:
                        _ = input('episode done, press Enter to replay')
                        epoch_memory = []
                        continue

                if len(epoch_memory) >= ROLLOUT_SIZE:
                    break

            # add to the global memory
            MEMORY.extend(epoch_memory)

            print('>>> Train phase')
            print('rollout reward: {}'.format(_rollout_reward))

            # Here we train only once.
            _, _global_step = sess.run([train_op, global_step])

            if _global_step % args.save_checkpoint_steps == 0:

                print('Writing summary')

                feed_dict = {rollout_reward: _rollout_reward}
                summary = sess.run(merged, feed_dict=feed_dict)

                summary_writer.add_summary(summary, _global_step)

                save_path = os.path.join(args.output_dir, 'model.ckpt')
                save_path = saver.save(sess, save_path, global_step=_global_step)
                print('Model checkpoint saved: {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('pong trainer')
    parser.add_argument(
        '--n-epoch',
        type=int,
        default=6000)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000)
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/pong_output')
    parser.add_argument(
        '--job-dir',
        type=str,
        default='/tmp/pong_output')

    parser.add_argument(
        '--restore',
        default=False,
        action='store_true')
    parser.add_argument(
        '--render',
        default=False,
        action='store_true')
    parser.add_argument(
        '--save-checkpoint-steps',
        type=int,
        default=1)

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-3)
    parser.add_argument(
        '--decay',
        type=float,
        default=0.99)
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99)
    parser.add_argument(
        '--laziness',
        type=float,
        default=0.01)
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=200)

    args = parser.parse_args()

    # save all checkpoints
    args.max_to_keep = args.n_epoch // args.save_checkpoint_steps

    main(args)
