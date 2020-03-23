import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import random
import numpy as np
from collections import namedtuple
from datetime import datetime
import os


class DQN(Model):
    """
    Q-Network
    Takes the encoded state and outputs the Q values for that state (expected total reward)
    """

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.d1 = Dense(10, activation="tanh")
        self.d2 = Dense(10, activation="tanh")
        self.d3 = Dense(2, activation="softmax")

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity     # Starts overwritting from 0 if full

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Bot(object):
    """
    Q-L
    """
    def __init__(self):
        # Constants
        self.num_actions = 2
        self.rewards = {0: 1.0, 1: -1000.0}
        self.lr = 0.99
        self.epsilon = 1
        self.epsilon_decay_steps = 500
        self.epsilon_decay = 0.9
        self.memory = ReplayMemory(10000)
        self.batch_size = 128
        self.update_target_eps = 50     # number of episodes between each update of target_model
        self.save_eps = 500

        # Variables
        self.step = 0  # Number of updates
        self.episode_start_step = 0
        self.episode_count = 0

        # Policy network
        inputs = tf.keras.Input(shape=(5,))
        outputs = DQN(self.num_actions)(inputs)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        target_outputs = DQN(self.num_actions)(inputs)
        self.target_model = tf.keras.Model(inputs=inputs, outputs=target_outputs)
        self.update_target_model()
        self.loss_fn = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.SGD(0.01)

        print(self.model.summary())

        # Logging
        now = datetime.now()
        self.logdir = os.path.join("summaries", now.strftime("%Y%m%d-%H%M%S"))
        self.summary_writer = tf.summary.create_file_writer(self.logdir)
        print("Saving summaries to {}".format(self.logdir))


    def update_target_model(self):
        """
        Copy weights from the policy network into the target network.
        """
        self.target_model.set_weights(self.model.get_weights())


    def gameover(self, score):
        """
        Update target network every update_target_eps completed episodes
        """

        # Log episode length
        episode_length = self.step-self.episode_start_step
        with self.summary_writer.as_default():
            tf.summary.scalar('episode_length', episode_length, self.episode_count)
        self.episode_start_step = self.step
        self.episode_count += 1

        # Update target network
        if self.episode_count % self.update_target_eps == 0:
            print("Updating target model")
            self.update_target_model()

        if self.episode_count % self.save_eps == 0:
            print("Saving model checkpoint")
            self.target_model.save_weights("{}/model_weights_{}.ckpt".format(self.logdir, str(self.episode_count)))


    def act(self, current_state):
        """
        Compute action to take based on current_state. Uses a e-greedy exploration policy

        Args:
            current_state (np.array): state encoding
        """
        # epsilon decay
        if self.step % self.epsilon_decay_steps == 0:
            self.epsilon = self.epsilon*self.epsilon_decay
            with self.summary_writer.as_default():
                tf.summary.scalar('epsilon', self.epsilon, step=self.step)

        self.step += 1

        batch = tf.expand_dims(current_state, axis = 0)

        # Compute action with max Q value for current_state
        q = self.model(batch)
        max_Q = np.argmax(q)

        # choose an action using e-greedy
        if random.random() <= self.epsilon:
            print("Acting randomly")
            action_index = random.randrange(2)
        else:
            action_index = max_Q

        return action_index


    def observe_transition(self, state, action, next_state, reward):
        """
        Store transition tuple (state, action, next_state, reward) in ReplayMemory

        Args:
            state (np.array): state encoding
            action (int): action taken
            next_state (np.array): state encoding
            reward (int): reward for transitioning into next_state
        """
        self.memory.push(state, action, next_state, reward)


    def update_model(self):
        """
        Sample a batch from the Replay Memory and update the policy network using Bellman Equation
        """
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
         # Transpose the batch (see https://stackoverflow.com/a/19343/3343043)
        batch = Transition(*zip(*transitions))
        state_batch = tf.stack(batch.state, axis=0)
        action_batch = tf.expand_dims(tf.stack(batch.action, axis=0), 1)
        reward_batch = tf.stack(batch.reward, axis=0)

        non_final_state_mask = tf.constant([s is not None for s in batch.next_state])
        next_state_batch = np.array([s if s is not None else np.zeros((5,)) for s in batch.next_state ], dtype=np.float32)

        model_variables = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_variables)

            state_qvalues = self.model(state_batch)    #Q(s_t,)   (30,2)
            state_action_qvalues = tf.gather_nd(state_qvalues, indices=action_batch, batch_dims=1) #Q(s_t, a)   (30,)

            next_state_batch = tf.constant(next_state_batch)
            next_state_values = tf.reduce_max(self.target_model(next_state_batch), 1)
            next_state_values = tf.where(non_final_state_mask, next_state_values, tf.zeros_like(next_state_values))

            # Bellman equation
            target_qvalues = reward_batch + self.lr * next_state_values # (30,)

            # Compute loss
            l = self.loss_fn(state_action_qvalues, target_qvalues)


            with self.summary_writer.as_default():
                tf.summary.scalar('loss', l, step=self.step)

            # Compute the gradients and update model
            gradients = tape.gradient(l, model_variables)
            self.optimizer.apply_gradients(zip(gradients, model_variables))

            # log gradient mean
            mean_grad = 0
            for g in gradients:
                mean_grad += tf.reduce_mean(g)

            mean_grad = tf.reduce_mean(mean_grad)
            with self.summary_writer.as_default():
                tf.summary.scalar('mean_gradient', mean_grad, step=self.step)








