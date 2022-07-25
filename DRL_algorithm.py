import tensorflow as tf
import numpy as np
import time
import json,ast
import math

from env_discrete_srcdst import Environment
import sys
# sys.path.append('./RoutingGeant/DRL/dRSIR/23nodos')
sys.path.insert(1,'/home/brayam/Tesis/DRL/AgentDRL/ddpg_disc_srcdst')
from ddpg_disc_srcdst.ddpg import DDDPGAgent

def get_all_paths(episode_rewards, episode_states_all, episode_duration_all, episodes, k_paths):
    t = time.time()
    env = Environment(step_per_epoch=40)
    state_space_size = env.obs_pm_size
    action_space_size = 1

    target_update_freq=100 #1000, #cada n steps se actualiza la target network
    discount=0.1
    batch_size = 32
    replay_memory_size = 5000#100000,
    # replay_start_size= 400
    # lr = 0.01

    agent = agent = DDDPGAgent(state_space_size, action_space_size, 1, 0, replay_memory_size, batch_size, 0.003)

    for episode in range(episodes):
        prev_state = env.reset()
        
        while True:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = agent.policy(tf_prev_state)
            # Recieve state and reward from environment.        
            
            state, reward, done, _ = env.step(action)        

            agent.buffer.record((prev_state, action[0], reward, state))
            # ep_reward_list.append(reward)
            if episode > 32:
                agent.learn()
                
                if episode % 1 == 0:            
                    agent.update_target(agent.target_actor.variables, agent.actor_model.variables)
                    agent.update_target(agent.target_critic.variables, agent.critic_model.variables)
            
            # # End this episode when `done` is True
            if done:
                break

            prev_state = state                  

    #Use trained model to find choosen actions
    drl_paths = {src: {dst: [] for dst in range(1,len(env.topo_nodes)+1) if src != dst} for src in range(1,len(env.topo_nodes)+1)}
    for src in range(1,len(env.topo_nodes)+1):
                for dst in range(1,len(env.topo_nodes)+1):
                    if src != dst:
                        state = [np.float32(env.obs_space.index((src,dst)))]
                        action = agent.policy(tf.expand_dims(tf.convert_to_tensor(state), 0))
                        action = continuosToDiscrete(action)
                        path = k_paths[str(src)][str(dst)][int(action)]
                        drl_paths[src][dst].append(path)

    #write choosen paths
    setPaths(drl_paths)
    # print('\t Time total: ',time.time()-t)
    # print("tup: {0} \ndisc: {1} \nbs: {2} \nminexp: {3} \nannr: {4} \nrms: {5} \nrss: {6} \nneu: {7}".format(agente.target_update_freq,agente.discount,agente.batch_size, agente.min_explore,agente.anneal_rate))#,agente.replay_memory_size,agente.replay_start_size))
    time_DRL = time.time() - t
    return time_DRL, episode_rewards, episode_states_all, episode_duration_all


def getKPaths():
    file = '/home/brayam/Tesis/Daniela/DRSIR-DRL-routing-approach-for-SDN/SDNapps_proac/RoutingGeant/DRL/dRSIR/23nodos/k_paths.json'
    with open(file,'r') as json_file:
        k_paths = json.load(json_file)
        return ast.literal_eval(json.dumps(k_paths))

def setPaths(drl_paths):
    with open('/home/brayam/Tesis/Daniela/DRSIR-DRL-routing-approach-for-SDN/SDNapps_proac/drl_paths.json','w') as json_file:
        json.dump(drl_paths, json_file, indent=2)

def continuosToDiscrete(actions_conti):        
    act_discrete = math.floor(actions_conti[0] * 20)        
    return 19 if act_discrete == 20  else act_discrete

if __name__ == "__main__":
    print('Start Agent')
    k_paths = getKPaths()

    cont = 0
    episodes = 40 
    epoch = 60
    # ---------TRAINNING AND RECOVERING OF PATHS----------
    # For running after deciding

    episode_rewards = [[] for _ in range(episodes)]
    episode_states_all = [[] for _ in range(episodes)]
    episode_duration_all = [[] for _ in range(episodes)]
    iteration_times = []
    while cont < epoch: 

        a = time.time()
        cont = cont + 1
        time_DRL, episode_rewards, episode_states_all, episode_duration_all = get_all_paths(episode_rewards, episode_states_all, 
                                                                                                        episode_duration_all, episodes, k_paths)

        iteration_times.append(time_DRL) # print('time_stretch' , time_stretch)
        sleep = 5 - time_DRL

        if sleep > 0:
            print("**"+str(cont)+"**time remaining drl and stretch",sleep)
            time.sleep(sleep)
        else:
            print("**"+str(cont)+"**time remaining drl and stretch",sleep)
            time.sleep(0.2)