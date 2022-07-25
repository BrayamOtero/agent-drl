import itertools
import random
import json, ast
import os
import numpy as np
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Environment(object):
    def __init__(self, step_per_epoch=30):
        
        self.act_space = [i for i in range(20)]
        self.topo_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.obs_space = [i for i in itertools.permutations(self.topo_nodes,2)]
        self.act_space_size = len(self.act_space) #number of actions per state
        self.obs_pm_size = len(random.choice(self.obs_space)) #number of parameters per state NOT ENCODE
        self.obs_pm_size = 1 #number of parameters per state ENCODE
        self.obs_space_size = len(self.obs_space) #number of states
        self.s = random.randrange(len(self.obs_space))
        self.cont_steps = 0
        self.next_states = self.zigZag()
        self.P = {state: {action: [] for action in range(self.act_space_size)} for state in range(self.obs_space_size)}
        self.rewards_dic = self.path_metrics_to_reward()
        for src in range(1,len(self.topo_nodes)+1):
            for dst in range(1,len(self.topo_nodes)+1):
                if src != dst:
                    state = self.obs_space.index((src,dst)) #state represented as te index in the array of obs

                    new_state, done = self.rand_next_state(state) # #next state for state
                    for action in range(self.act_space_size):
                        reward =  self.rewards_dic[str(src)][str(dst)][action]
                        self.P[state][action].append((new_state, reward, done))
        
        self.tm = self.normalizeTM(0, 1)
        self.step_per_epoch = step_per_epoch

    def zigZag(self):
        # a=time.time()
        '''Given an array of DISTINCT elements, rearrange the elements
        of array in zig-zag fashion in O(n) time.
        return a < b > c < d > e < f
        Flag true if relation <, else ">" is expected.
        The first expected relation is "<" '''
        states = [i for i in range(self.obs_space_size)]
        flag = True
        for i in range(self.obs_space_size-1):
            # "<" relation expected
            if flag is True:
                # If we have a situation like A > B > C,
                #   we get A > B < C
                # by swapping B and C
                if states[i] > states[i+1]:
                    states[i], states[i+1] = states[i+1],states[i]
                # ">" relation expected
            else:
                # If we have a situation like A < B < C,
                #   we get A < C > B
                # by swapping B and C
                if states[i] < states[i+1]:
                    states[i],states[i+1] = states[i+1],states[i]
            flag = bool(1 - flag)
        next_states = states
        return next_states

    def rand_next_state(self, state):
        done = False

        #automatic all states random
        if self.next_states.index(state) == self.obs_space_size-1:
          next_state = self.next_states[0]
        else:
          next_state = self.next_states[self.next_states.index(state)+1]
        return next_state, done

    #------Reward------------
    def path_metrics_to_reward(self):

        # #Reads metrics paths file
        # ini = time.time()
        file = '/home/brayam/Tesis/Daniela/DRSIR-DRL-routing-approach-for-SDN/SDNapps_proac/paths_metrics.json'
        num_actions = self.act_space_size
        rewards_dic = {}
        metrics = ['bwd_paths','delay_paths','loss_paths', 'qlen_paths']


        with open(file,'r') as json_file:
            paths_metrics_dict = json.load(json_file)
            paths_metrics_dict = ast.literal_eval(json.dumps(paths_metrics_dict))

        for i in paths_metrics_dict:
            rewards_dic.setdefault(i,{})
            for j in paths_metrics_dict[i]:
                rewards_dic.setdefault(j,{})
                for m in metrics:
                    met_norm = [self.normalize(met_val, 0, 100, min(paths_metrics_dict[str(i)][str(j)][m][0]), max(paths_metrics_dict[str(i)][str(j)][m][0])) for met_val in paths_metrics_dict[str(i)][str(j)][m][0]]
                    paths_metrics_dict[str(i)][str(j)][m].append(met_norm)

        for i in paths_metrics_dict:
            for j in paths_metrics_dict[i]:
                rewards_actions = []
                for act in range(num_actions):
                    rewards_actions.append(self.reward(i,j,paths_metrics_dict,act,metrics))
                    rewards_dic[i][j] = rewards_actions

        return rewards_dic

    def reward(self, src, dst, paths_metrics_dict, act, metrics):
        '''
        paths_metrics_dict ={src:{dst:{metric1:[[orig value list],[normalized value list]]},metric2...}}
        '''
        beta1=1
        beta2=1
        beta3=1
        beta4=1
        cost_action=1
        reward = cost_action + beta1*paths_metrics_dict[str(src)][str(dst)][metrics[0]][1][act] - beta2*paths_metrics_dict[str(src)][str(dst)][metrics[1]][1][act] - beta3*paths_metrics_dict[str(src)][str(dst)][metrics[2]][1][act] - beta4*paths_metrics_dict[str(src)][str(dst)][metrics[3]][1][act]
        return round(reward,15)

    def normalize(self, value, minD, maxD, min_val, max_val):
        if max_val == min_val:
            value_n = (maxD + minD) / 2
        else:
            value_n = (maxD - minD) * (value - min_val) / (max_val - min_val) + minD
        return round(value_n,15)

    def reset(self):        
        self.s = random.randrange(len(self.obs_space))

        return self.getState(self.s)

    def step(self, a):
        self.cont_steps += 1
        a_disct = self.continuosToDiscrete(a)
        s, r, d = self.P[int(self.s)][int(a_disct)][0]

        self.s = s
        d = self.cont_steps == self.step_per_epoch
        if d:
            self.cont_steps = 0
        return (self.getState(self.s), r, d, '')

    def getState(self, s):
        return np.concatenate([self.tm.flatten(), np.array([s])])

    def normalizeTM(self,a,b):
        """Funcion para trandormar json a matix y normalizar la matrix de trafico la cual es recibida por un  archivo JSON
            Se normaliza los valores utilizando la siguiente funcion:

                norm_x_i = a + ((x_i- min(X))*(b-a))/(max(X) - min(X))
            
            Parameters
            ----------
            a : int
                el valor minimo a normalizar
            b : int
                valor maximo a normalizar
            TM_path: string
                el path donde el esta la TM en formato JSON
            
            Returns
            -------
            matrix np
                La matriz de trafico normalizada
        """

        file = '/home/brayam/Tesis/Daniela/DRSIR-DRL-routing-approach-for-SDN/SDNapps_proac/TM.json'

        with open(file,'r') as json_file:
            data = json.load(json_file)
            data = ast.literal_eval(json.dumps(data))        

        tm = np.zeros((23,23))

        for src, dst_bw in data.items():
            for dst, bw in dst_bw.items():
                tm[int(src)-1][int(dst)-1] = bw    
        bmax, bmin = tm.max(), tm.min()        
        # print("Tiempo necesario para tranformar y normalizar la TM {} ms".format(total_time))
        if bmax == bmin:
            return tm
        else:
            return a + ((tm-bmin)*(b-a)/(bmax-bmin))

    def continuosToDiscrete(self, actions_conti):        
        act_discrete = math.floor(actions_conti[0] * 20)        
        return 19 if act_discrete == 20  else act_discrete