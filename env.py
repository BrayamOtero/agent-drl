import numpy as np
import json, ast
import time

class Env:
    def __init__(self, sleep = 5):
        self.k_paths = self.retriveData('/home/brayam/Tesis/DRL/AgentDRL/k_paths.json')
        self.sleep = sleep        
        self.reward = self.retriveData('/home/brayam/Tesis/Daniela/DRSIR-DRL-routing-approach-for-SDN/SDNapps_proac/paths_metrics.json',)
        # mapea los enlaces a indice del vector de las acciones
        self.linksMapIndex = {(1, 3): 1, (3, 1): 1, (1, 7): 2, (7, 1): 2, (1, 16): 3, (16, 1): 3, (2, 4): 4, (4, 2): 4, (2, 7): 5, (7, 2): 5, (2, 12): 6, (12, 2): 6, (2, 13): 7, (13, 2): 7, (2, 18): 8, (18, 2): 8, (2, 23): 9, (23, 2): 9, (3, 10): 10, (10, 3): 10, (3, 11): 11, (11, 3): 11, (3, 14): 12, (14, 3): 12, (3, 21): 13, (21, 3): 13, (4, 16): 14, (16, 4): 14, (5, 8): 15, (8, 5): 15, (5, 16): 16, (16, 5): 16, (6, 7): 17, (7, 6): 17, (6, 19): 18, (19, 6): 18, (7, 17): 19, (17, 7): 19, (7, 19): 20, (19, 7): 20, (7, 21): 21, (21, 7): 21, (8, 9): 22, (9, 8): 22, (9, 15): 23, (15, 9): 23, (9, 16): 24, (16, 9): 24, (10, 11): 25, (11, 10): 25, (10, 12): 26, (12, 10): 26, (10, 16): 27, (16, 10): 27, (10, 17): 28, (17, 10): 28, (12, 22): 29, (22, 12): 29, (13, 14): 30, (14, 13): 30, (13, 17): 31, (17, 13): 31, (13, 19): 32, (19, 13): 32, (15, 20): 33, (20, 15): 33, (17, 20): 34, (20, 17): 34, (17, 23): 35, (23, 17): 35, (18, 21): 36, (21, 18): 36, (20, 22): 37, (22, 20): 37}                    

    def retriveData(self, path):
        with open(path, 'r') as json_file:
            data = json.load(json_file)
            data = ast.literal_eval(json.dumps(data)) 
        return data

    def step(self, actions):
        '''
            Envia el path de las tuplas src dst, y espera un tiempo determinado (5 segunos que son el tiempo de monitorizaciòn)
            para que el controlador instale las rutas en el plano de datos.

            Parameters
            ----------
            actions : array(int)
                array con los pesos de los enlaces

            Returns
            -------
            next_state np
                El siguiente estado, que este caso es la matrix de tràfico normalizada
            reward int
                La recompensa recibida, que es la suma del bw disponible + loss + delay + qlen

        '''
        actions_paths = self.actionToPaths(actions)        
        # self.sendPaths(actions_paths)
        # time.sleep(self.sleep)
        return self.normalizeState(0,1), self.normalizeReward(actions_paths)

    def sendPaths(self, paths):
        paths_json = {}    

        for src, dsts in paths.items():
            paths_json.setdefault(src, {})
            for dst, idx_path in dsts.items():
                paths_json[src][dst] = [self.k_paths[src][dst][idx_path-1]]
        
        json_file = json.dumps(paths_json, indent=4)
        with open("/home/brayam/Tesis/Daniela/DRSIR-DRL-routing-approach-for-SDN/SDNapps_proac/drl_paths.json", "w") as outfile:
            outfile.write(json_file)        

    def normalizeState(self,a,b):
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
        i = time.time()

        data = self.retriveData('/home/brayam/Tesis/Daniela/DRSIR-DRL-routing-approach-for-SDN/SDNapps_proac/TM.json')

        tm = np.zeros((23,23))

        for src, dst_bw in data.items():
            for dst, bw in dst_bw.items():
                tm[int(src)-1][int(dst)-1] = bw    
        bmax, bmin = tm.max(), tm.min()        
        total_time = round((time.time() - i)*1000,3)
        # print("Tiempo necesario para tranformar y normalizar la TM {} ms".format(total_time))
        if bmax == bmin:
            return tm
        else:
            return a + ((tm-bmin)*(b-a)/(bmax-bmin))        

    def normalizeReward(self, actions_paths):        
        i = time.time()            
        rewards_dic = {}
        metrics = ['bwd_paths','delay_paths','loss_paths', 'qlen_paths']

        bw_arr = []
        delay_arr = []
        loss_arr = []
        qlen_arr = []
        
        # data = self.retriveData('/home/brayam/Tesis/Daniela/DRSIR-DRL-routing-approach-for-SDN/SDNapps_proac/paths_metrics.json',) 
        data = self.reward       
        
        for src, dsts in data.items():
            rewards_dic.setdefault(src,{})
            for dst in dsts.keys():
                rewards_dic[src].setdefault(dst,{})
                for m in metrics:                
                    index = actions_paths[src][dst]                
                    val = data[str(src)][str(dst)][m][0][index-1]                                
                    rewards_dic[str(src)][str(dst)][m] = val

                    if m == metrics[0]:
                        bw_arr.append(val) 
                        # if val > 0.001: #ensure minimum bwd available
                        #     rewards_dic[str(src)][str(dst)][m] = round(1/val, 15)
                        #     bw_arr.append(round(1/val, 15))                            
                        # else:                                               
                        #     rewards_dic[str(src)][str(dst)][m] = 1/0.001
                        #     bw_arr.append(1/0.001)
                    elif m == metrics[1]:                    
                        delay_arr.append(val)
                    elif m == metrics[2]:                    
                        loss_arr.append(val)
                    elif m == metrics[3]:                    
                        qlen_arr.append(val)
    
        minbw, maxbw = min(bw_arr), max(bw_arr)
        mindelay, maxdelay = min(delay_arr), max(delay_arr)
        minloss, maxloss = min(loss_arr), max(loss_arr)
        minqlen, maxqlen = min(qlen_arr), max(qlen_arr)

        bwT, delayT, lossT, qlenT = 0,0,0,0

        for src, dsts in rewards_dic.items():
            for dst, mets in dsts.items():            
                for m in metrics:                
                    if m == metrics[0]:
                        # rewards_dic[src][dst][m] = normalize(mets[m], 0,100, minbw, maxbw)
                        bwT = self.normalize(mets[m], 0,100, minbw, maxbw)                                         
                    elif m == metrics[1]:
                        # rewards_dic[src][dst][m] = normalize(mets[m], 0,100, mindelay, maxdelay)
                        delayT = self.normalize(mets[m], 0,100, mindelay, maxdelay)
                    elif m == metrics[2]:
                        # rewards_dic[src][dst][m] = normalize(mets[m], 0,100, minloss, maxloss)
                        lossT =  self.normalize(mets[m], 0,100, minloss, maxloss)
                    elif m == metrics[3]:
                        # rewards_dic[src][dst][m] = normalize(mets[m], 0,100, minqlen, maxqlen)
                        qlenT = self.normalize(mets[m], 0,100, minqlen, maxqlen)

        
        rewardT = bwT - delayT - lossT - qlenT

        print("La recompensa es:  ",rewardT)
        total_time = round((time.time() - i)*1000,3)
        # print("Tiempo necesario para normalizar la reward {} ms".format(total_time))

        return rewardT

    def normalize(self, value, minD, maxD, min_val, max_val):
        if max_val == min_val:
            value_n = (maxD + minD) / 2
        else:
            value_n = (maxD - minD) * (value - min_val) / (max_val - min_val) + minD
        return round(value_n,15)

    def actionToPaths(self, arrWeigth):    
        
        i = time.time()
        
        # print(arrWeigth[0][2])
        sum_all_paths = {}
        for src, dsts in self.k_paths.items():
            sum_all_paths[src] = {}        
            for dst, paths in dsts.items():            
                arr_sum_path = []
                
                for path in paths:
                    source = 0
                    sum_path = 0

                    for node in path:
                        if source == 0:
                            source = node
                        else:
                            sum_path += arrWeigth[0][self.linksMapIndex[(source, node)] - 1]
                            source = node 
                    arr_sum_path.append(sum_path)            
                sum_all_paths[src][dst] = arr_sum_path.index(min(arr_sum_path)) + 1 
        total_time = round((time.time() - i)*1000,3)
        # print("Tiempo necesario para tranformar y normalizar la TM {} ms".format(total_time))
        return sum_all_paths