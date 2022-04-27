import ddpg
import tensorflow as tf
import numpy as np
from ddpg import DDDPGAgent

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

agent = DDDPGAgent(23*23, 37, 1, 0, 50000, 64)
total_episodes = 1

# Takes about 4 min to train
for ep in range(total_episodes):

    prev_state = np.zeros([23,23])
    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state.flatten()), 0)

        action = agent.policy(tf_prev_state)
        # Recieve state and reward from environment.
        state, reward = np.zeros([23,23]).flatten(), 1
        # print(action)

        agent.buffer.record((prev_state.flatten(), action[0], reward, state.flatten()))
        episodic_reward += reward

        agent.learn()
        agent.update_target(agent.target_actor.variables, agent.actor_model.variables)
        agent.update_target(agent.target_critic.variables, agent.critic_model.variables)

        # # End this episode when `done` is True
        # if done:
        #     break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
