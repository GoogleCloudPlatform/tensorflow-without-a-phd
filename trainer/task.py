import argparse
import os
import tensorflow as tf
import numpy as np
import gym
import json

from helpers import discount_rewards, prepro

# Open AI gym Atari env: 0: 'NOOP', 2: 'UP', 3: 'DOWN'
NOOP_ACTIONS = [0, 2, 3]
ACTIONS = [2, 3]

OBSERVATION_DIM = 80 * 80

def main(args):
    args_dict = vars(args)

    args_string = json.dumps(args_dict)
    args_tensor = tf.constant(args_string)
    args_summary = tf.summary.tensor_summary('args', args_tensor)

    print('args: {}'.format(args_dict))

    tf.reset_default_graph()
    observations = tf.placeholder(shape=(None, OBSERVATION_DIM), dtype=tf.float32)

    layers = [observations]

    for hidden_dim in args.hidden_dims:
        _input = layers[-1]
        _output = tf.layers.dense(
            inputs=_input,
            units=hidden_dim,
            use_bias=False,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        layers.append(_output)

    logits = tf.layers.dense(
        inputs=layers[-1],
        units=len(args.actions),
        use_bias=False,
        # linear activation
        activation=None,
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)
    )

    logits_for_sampling = tf.reshape(logits, shape=(1, len(args.actions)))
    sample_action = tf.multinomial(logits=logits_for_sampling, num_samples=1)

    # adding additional cost to actions other than NOOP
    action_costs_tensor = args.beta * tf.constant(args.action_costs, dtype=tf.float32)
    probs = tf.nn.softmax(logits=logits)

    # Note: The extra loss can be interpreted terms as KL divergence.
    # Consider Q = (q0, q1, q2) the estimated probs.
    # Consider P = (0, cq1, cq2), where c is a normalizing constant.
    # Where 1 = c(q1+q2) = c(1-q0)
    # Then KL(P||Q) = c(q1+q2)log(c) = clog(c) * extra_loss.
    # Note that KL is bounded between 0 and 1, c >= 0, and clog(c) is
    # strictly increacing for c >= 0, and c and q0 increase together.
    extra_loss = tf.reduce_sum(probs * action_costs_tensor)

    labels = tf.placeholder(
        shape=(None, ),
        dtype=tf.int32
    )
    rewards = tf.placeholder(
        shape=(None, ),
        dtype=tf.float32
    )
    processed_rewards = tf.placeholder(
        shape=(None, ),
        dtype=tf.float32
    )

    cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    loss = tf.reduce_sum(processed_rewards * cross_entropies) + extra_loss

    n_episode_played = tf.placeholder(shape=(), dtype=tf.float32)

    batch_reward = tf.reduce_sum(rewards) / n_episode_played

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=args.learning_rate,
        decay=args.decay
    )
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=args.learning_rate
    )
    train_op = optimizer.minimize(loss)

    # global_step records the number of times the weights are updated
    global_step = tf.Variable(0, trainable=False, name='global_step')
    increment_global_step = tf.assign(global_step, global_step + 1)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)

    # The weights of the first hidden layer can be visualized.
    first_hidden = layers[1]
    hidden_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(first_hidden.name)[0] + '/kernel:0')

    for h in xrange(args.hidden_dims[0]):
        slice_ = tf.slice(hidden_weights, [0, h], [-1, 1])
        image = tf.reshape(slice_, [1, 80, 80, 1])
        tf.summary.image('hidden_{:04d}'.format(h), image)

    tf.summary.scalar('batch_reward', batch_reward)
    tf.summary.scalar('loss', loss)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        if args.restore:
            restore_path = tf.train.latest_checkpoint(args.output_dir)
            print('Restoring from {}'.format(restore_path))
            saver.restore(sess, restore_path)
        else:
            sess.run(init)

        env = gym.make("Pong-v0")

        if not args.dry_run:
            summary_path = os.path.join(args.output_dir, 'summary')
            summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

        for i in range(args.n_batch):
            _observations = []
            _labels = []
            _rewards = []
            _processed_rewards = []
            _n_episode_played = 0.0

            for j in range(args.batch_size):
                print('>>>>>>> {} / {} of batch {}'.format(j+1, args.batch_size, i))
                state = env.reset()
                previous_x = None
                _episode_rewards = []
                _episode_preocessed_rewards = []

                # The loop for actions/steps
                while True:
                    if args.render:
                        env.render()

                    current_x = prepro(state)
                    _observation = current_x - previous_x if previous_x is not None else np.zeros(OBSERVATION_DIM)
                    previous_x = current_x

                    # sample one action with the given probability distribution
                    _label = int(sess.run(sample_action, feed_dict={observations: [_observation]})[0, 0])

                    _action = args.actions[_label]

                    state, reward, done, info = env.step(_action)

                    _observations.append(_observation)
                    _labels.append(_label)
                    _episode_rewards.append(reward)

                    if done:
                        _n_episode_played += 1
                        break

                # Process the rewards after each espisode
                _episode_preocessed_rewards = discount_rewards(_episode_rewards, args.gamma)
                _episode_preocessed_rewards -= np.mean(_episode_preocessed_rewards)
                _episode_preocessed_rewards /= np.std(_episode_preocessed_rewards)

                _rewards.extend(_episode_rewards)
                _processed_rewards.extend(_episode_preocessed_rewards)

                if len(_rewards) >= args.max_steps:
                    print('Max steps reached.')
                    break

            _observations = np.array(_observations)
            _labels = np.array(_labels)
            _rewards = np.array(_rewards)

            feed_dict = {
                observations: _observations,
                labels: _labels,
                rewards: _rewards,
                processed_rewards: _processed_rewards,
                n_episode_played: _n_episode_played,
            }

            assert len(_observations) == len(_labels)
            assert len(_labels) == len(_rewards)
            assert len(_rewards) == len(_processed_rewards)

            if not args.dry_run:
                g_step = sess.run(global_step)
                if g_step % args.save_summary_steps == 0:
                    print('Writing summary')
                    summary = sess.run(merged, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, g_step)

                print('Updating weights')
                _ = sess.run([train_op, increment_global_step], feed_dict=feed_dict)

                if g_step % args.save_checkpoint_steps == 0:
                    save_path = os.path.join(args.output_dir, 'model.ckpt')
                    save_path = saver.save(sess, save_path, global_step=g_step)
                    print('Model checkpoint saved: {}'.format(save_path))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('pong trainer')
    parser.add_argument(
        '--n-batch',
        type=int,
        default=6000)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10)
    parser.add_argument(
        '--max-steps',
        type=int,
        default=20000,
        help='Maximum number of steps before weight update.')
    parser.add_argument(
        '--save-summary-steps',
        type=int,
        default=20)
    parser.add_argument(
        '--save-checkpoint-steps',
        type=int,
        default=40)
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
        '--dry-run',
        default=False,
        action='store_true')

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--decay',
        type=float,
        default=0.99)
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99)
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[200])
    parser.add_argument(
        '--allow-noop',
        default=False,
        action='store_true')
    parser.add_argument(
        '--action-costs',
        type=int,
        nargs='+',
        default=[0, 0])
    parser.add_argument(
        '--beta',
        type=float,
        default=0.01)

    args = parser.parse_args()

    # save all checkpoints!
    args.max_to_keep = args.n_batch / args.save_checkpoint_steps

    if args.allow_noop:
        args.actions = NOOP_ACTIONS
        args.action_costs = [0, 1, 1]
    else:
        args.actions = ACTIONS
        args.action_costs = [0, 0]

    main(args)
