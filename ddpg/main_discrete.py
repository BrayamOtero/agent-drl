import sys
import time

import tensorflow as tf
import numpy as np
from ddpg import DDDPGAgent

sys.path.append("..")
from env_discrete import Environment

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

#El estado del agente sera la TM concatenada con la tupla de nodes(src, dst), por tal motivo el numero de acciones estados es:
# 23*23 + 1 donde 23 es la cantidad de nodos.
#Esto con el fin de en contrar el path desde el src hasta dst de cada una de las tuplas
#Sin necesidad de obtener los pesos de los enlaces, sino que el agente entregara directamente de el path de la tupla
# de un conjunto de k paths, donde k = 20.
# TODO: Como ddpg es de acciones continuas, por lo tanto se van a discretizar, el dominio del esapcio de acciones es 
# entre 0 y 1, los cuales se van a dividir en 20 puntos equidistantes.
num_state = (23 * 23) + 1
agent = DDDPGAgent(num_state, 1, 1, 0, 50000, 64)
env = Environment()
total_episodes = 300

# Takes about 4 min to train
i = time.time()
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0
    #reset env    

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = agent.policy(tf_prev_state)
        # Recieve state and reward from environment.        
        
        state, reward, done, _ = env.step(action)        

        agent.buffer.record((prev_state, action[0], reward, state))        
        
        ep_reward_list.append(reward)

        agent.learn()
        agent.update_target(agent.target_actor.variables, agent.actor_model.variables)
        agent.update_target(agent.target_critic.variables, agent.critic_model.variables)

        # # End this episode when `done` is True
        if done:
            break

        prev_state = state        

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list)
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)
    ep_reward_list = []

total_time = round((time.time() - i), 3)

# guardar los pesos de los modelos
# agent.actor_model.save('weigths/actor_model.h5')
# agent.critic_model.save('weigths/critic_model.h5')

# agent.target_actor.save('weigths/target_actor_model.h5')
# agent.target_critic.save('weigths/target_critic_model.h5')

print("Tiempo del proceso: {} s.".format(total_time))