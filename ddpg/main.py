import os, sys
import time

import tensorflow as tf
import numpy as np
from ddpg import DDDPGAgent

sys.path.append("..")
from env import Env

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

agent = DDDPGAgent(23*23, 37, 1, 0, 50000, 64)
env = Env()
total_episodes = 10

# Takes about 4 min to train
i = time.time()
for ep in range(total_episodes):

    prev_state = np.zeros([23,23])
    episodic_reward = 0
    #reset env
    env.reset()

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state.flatten()), 0)

        action = agent.policy(tf_prev_state)
        # Recieve state and reward from environment.        
        
        state, reward, done = env.step(action)        

        agent.buffer.record((prev_state.flatten(), action[0], reward, state.flatten()))
        episodic_reward += reward

        agent.learn()
        agent.update_target(agent.target_actor.variables, agent.actor_model.variables)
        agent.update_target(agent.target_critic.variables, agent.critic_model.variables)

        # # End this episode when `done` is True
        if done:
            break

        prev_state = state

        ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list)
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)
    ep_reward_list = []

total_time = round((time.time() - i), 3)

# guardar los pesos de los modelos
agent.actor_model.save('weigths/actor_model.h5')
agent.critic_model.save('weigths/critic_model.h5')

agent.target_actor.save('weigths/target_actor_model.h5')
agent.target_critic.save('weigths/target_critic_model.h5')

print("Tiempo del proceso: {} s.".format(total_time))