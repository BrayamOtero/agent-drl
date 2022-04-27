import tensorflow as tf
import numpy as np

from models import *
from buffer import Buffer
from ouActionNoise import OUActionNoise

class DDDPGAgent:
    def __init__(self, num_states,
                    num_actions,
                    upper_bound,
                    lower_bound,
                    buffer_capacity,
                    batch_size,
                    std_dev = 0.2,
                    critic_lr = 0.002,
                    actor_lr = 0.001,
                    total_episodes = 100,
                    gamma = 0.99,
                    tau = 0.005):
        self.num_states = num_states
        self.num_actions = num_actions
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.std_dev = std_dev
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.total_episodes = total_episodes
        self.gamma = gamma
        self.tau = tau
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.actor_model = Actor(self.num_states, self.num_actions, self.upper_bound).getModel()
        self.critic_model = Critic(self.num_states, self.num_actions).getModel()

        self.target_actor = Actor(self.num_states, self.num_actions, self.upper_bound).getModel()
        self.target_critic = Critic(self.num_states, self.num_actions).getModel()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        self.buffer = Buffer(num_states, num_actions, self.buffer_capacity, self.batch_size)

        self.ou_noise = OUActionNoise(mean = np.zeros(num_actions), std_deviation=float(self.std_dev)*np.ones(num_actions))


    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = self.ou_noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]

    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)            
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample()

        # Convert to tensors
        state_batch = tf.convert_to_tensor(state_batch)
        action_batch = tf.convert_to_tensor(action_batch)
        reward_batch = tf.convert_to_tensor(reward_batch)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch)

        self.update(state_batch, action_batch, reward_batch, next_state_batch)
