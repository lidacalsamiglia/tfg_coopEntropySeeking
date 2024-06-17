import environment as env
from copy import deepcopy
from math import log
import numpy as np
import pickle

########################################
###### Collaborative MOP super-agent ###
########################################
class SuperAgentHCollab:
    def __init__(self, s01,s02, u01,u02, arena, params):
        self.s01 = s01
        self.s02 = s02
        self.u01 = u01
        self.u02 = u02
        self.arena = arena
        self.params = params
    def h_iteration(self, tolerance, n_iter, verbose):
        value = np.zeros((self.arena.sizex, self.arena.sizey, self.arena.sizex, self.arena.sizey, self.arena.sizeu, self.arena.sizeu))
        print("sizeu", self.arena.sizeu)
        #value_old = deepcopy(value)
        value_old = np.copy(value)
        t_stop = n_iter
        for t in range(n_iter):
            ferror_max = 0
            for u1 in range(self.arena.sizeu):
                for u2 in range(self.arena.sizeu):
                    for x1 in range(self.arena.sizex):
                        for y1 in range(self.arena.sizey):
                            for x2 in range(self.arena.sizex):
                                for y2 in range(self.arena.sizey):
                                    s1 = [x1,y1]
                                    s2 = [x2,y2]
                                    actions1, _ = env.adm_actions(s1, u1, self.arena, self.params)
                                    actions2, _ = env.adm_actions(s2, u2, self.arena, self.params)
                                    Z = 0
                                    for a1 in actions1:
                                        for a2 in actions2:
                                            s_primes1 = env.transition_s(s1, a1, self.arena)
                                            s_primes2 = env.transition_s(s2, a2, self.arena)
                                            expo = 0
                                            for s_prime1 in s_primes1:
                                                for s_prime2 in s_primes2:
                                                    u_prime1 = env.transition_uv1(s1, s2, u1, self.arena)
                                                    u_prime2 = env.transition_uv1(s2, s1, u2, self.arena)
                                                    expo += self.params.gamma * value_old[s_prime1[0], s_prime1[1], s_prime2[0], s_prime2[1],u_prime1, u_prime2]
                                            Z += np.exp(expo / self.params.alpha)   
                                    value[x1,y1,x2,y2,u1,u2] = self.params.alpha * log(Z)
                                    f_error = abs( value[x1, y1,x2,y2, u1,u2] - value_old[x1, y1,x2,y2, u1,u2])
                                    ferror_max = max(ferror_max, f_error)
            if ferror_max < tolerance:
                t_stop = t
                break
            if verbose:
                print(f"iteration = {t}, max function error = {ferror_max}")
            #value_old = deepcopy(value)
            value_old = np.copy(value)
            with open('hvalue_foodIFbothin.pkl', 'wb') as f:
                pickle.dump(value_old, f)
                print(f"value at iter {t} saved")
        return value, t_stop
    
    def optimal_policy(self, s1, s2, u1, u2, optimal_value, verbose):
        actions1, _ = env.adm_actions(s1, u1, self.arena, self.params)
        actions2, _ = env.adm_actions(s2, u2, self.arena, self.params)
        policy = np.zeros(len(actions1) * len(actions2))
        # Only compute policy for available actions
        for idx1, a1 in enumerate(actions1):
            for idx2, a2 in enumerate(actions2):
                u_p1 = env.transition_uv1(s1, s2, u1, self.arena)
                u_p2 = env.transition_uv1(s2, s1, u2, self.arena)
                s_p1 = env.transition_s(s1, a1, self.arena)[0]
                s_p2 = env.transition_s(s2, a2, self.arena)[0]
                prob = np.exp(self.params.gamma*optimal_value[s_p1[0], s_p1[1], s_p2[0], s_p2[1], u_p1, u_p2]/self.params.alpha
                                            - optimal_value[s1[0], s1[1], s2[0], s2[1], u1, u2]/self.params.alpha)        
                policy[idx1 * len(actions2) + idx2] = prob
        # adjust for numerical errors in probability
        policy = policy / sum(policy)
        if verbose:
            print("ACTIONS1: ", actions1)
            print("ACTIONS2: ", actions1)
            print("state = ", s1, s2, " u = ", u1, u2)
            print("policy = ", policy)
        # return list of actions and list of probabilities
        return (actions1, actions2), policy
    
    def sample_trajectory(self,s1_0,s2_0, u1_0, u2_0, opt_value, max_t):
        xpositions1 = []
        xpositions2 = []
        ypositions1 = []
        ypositions2 = []
        u_states1 = []
        u_states2 = []
        
        xpositions1.append(s1_0[0])
        ypositions1.append(s1_0[1])
        xpositions2.append(s2_0[0])
        ypositions2.append(s2_0[1])
       
        u_states1.append(u1_0)
        u_states2.append(u2_0)
       
        s1 = s1_0.copy()
        s2 = s2_0.copy()
        u1 = u1_0
        u2 = u2_0
        value = np.zeros(max_t)
        for t in range(max_t):
            (actions1,actions2), policy = self.optimal_policy(s1, s2, u1, u2, opt_value, False)
            idx = np.random.choice(len(policy), p=policy)
            action1 = actions1[idx // len(actions2)]
            action2 = actions2[idx % len(actions2)]
            u1= env.transition_uv1(s1, s2, u1, self.arena)
            u2 = env.transition_uv1(s2, s1, u2, self.arena)
            s1 = env.transition_s(s1, action1, self.arena)[0]
            s2 = env.transition_s(s2, action2, self.arena)[0]
            xpositions1.append(s1[0])
            ypositions1.append(s1[1])
            xpositions2.append(s2[0])
            ypositions2.append(s2[1])
            u_states1.append(u1)
            u_states2.append(u2)
            reward = -np.log(policy[idx])
            if t == 0:
                value[t] = reward
            else: 
                value[t] = value[t-1] + self.params.gamma**t * reward

        return xpositions1, ypositions1, u_states1, xpositions2, ypositions2, u_states2, value
    
###############################################
###### Non-collaborative MOP super-agent ######
###############################################
class SuperAgentHNonCollab:
    def __init__(self, s01,s02, u01,u02, arena, params):
        self.s01 = s01
        self.s02 = s02
        self.u01 = u01
        self.u02 = u02
        self.arena = arena
        self.params = params
    def h_iteration(self, tolerance, n_iter, verbose):
        value = np.zeros((self.arena.sizex, self.arena.sizey, self.arena.sizex, self.arena.sizey, self.arena.sizeu, self.arena.sizeu))
        print("sizeu", self.arena.sizeu)
        #value_old = deepcopy(value)
            
        value_old = np.copy(value)
        t_stop = n_iter
        for t in range(n_iter):
            ferror_max = 0
            for u1 in range(self.arena.sizeu):
                for u2 in range(self.arena.sizeu):
                    for x1 in range(self.arena.sizex):
                        for y1 in range(self.arena.sizey):
                            for x2 in range(self.arena.sizex):
                                for y2 in range(self.arena.sizey):
                                    s1 = [x1,y1]
                                    s2 = [x2,y2]
                                    actions1, _ = env.adm_actions(s1, u1, self.arena, self.params)
                                    actions2, _ = env.adm_actions(s2, u2, self.arena, self.params)
                                    Z = 0
                                    for a1 in actions1:
                                        for a2 in actions2:
                                            s_primes1 = env.transition_s(s1, a1, self.arena)
                                            s_primes2 = env.transition_s(s2, a2, self.arena)
                                            expo = 0
                                            for s_prime1 in s_primes1:
                                                for s_prime2 in s_primes2:
                                                    u_prime1 = env.transition_u_one(s1, u1, self.arena)
                                                    u_prime2 = env.transition_u_one(s2, u2, self.arena)
                                                    expo += self.params.gamma * value_old[s_prime1[0], s_prime1[1], s_prime2[0], s_prime2[1],u_prime1, u_prime2]
                                            Z += np.exp(expo / self.params.alpha)   
                                    value[x1,y1,x2,y2,u1,u2] = self.params.alpha * log(Z)
                                    f_error = abs( value[x1, y1,x2,y2, u1,u2] - value_old[x1, y1,x2,y2, u1,u2])
                                    ferror_max = max(ferror_max, f_error)
            if ferror_max < tolerance:
                t_stop = t
                break
            if verbose:
                print(f"iteration = {t}, max function error = {ferror_max}")
            #value_old = deepcopy(value)
            value_old = np.copy(value)
            with open('hvalue_noncollab.pkl', 'wb') as f:
                pickle.dump(value_old, f)
        return value, t_stop
    
    def optimal_policy(self, s1, s2, u1, u2, optimal_value, verbose):
        actions1, _ = env.adm_actions(s1, u1, self.arena, self.params)
        actions2, _ = env.adm_actions(s2, u2, self.arena, self.params)
        policy = np.zeros(len(actions1) * len(actions2))
        # Only compute policy for available actions
        for idx1, a1 in enumerate(actions1):
            for idx2, a2 in enumerate(actions2):
                u_p1 = env.transition_u_one(s1, u1, self.arena)
                u_p2 = env.transition_u_one(s2, u2, self.arena)
                s_p1 = env.transition_s(s1, a1, self.arena)[0]
                s_p2 = env.transition_s(s2, a2, self.arena)[0]
                prob = np.exp(self.params.gamma*optimal_value[s_p1[0], s_p1[1], s_p2[0], s_p2[1], u_p1, u_p2]/self.params.alpha
                                            - optimal_value[s1[0], s1[1], s2[0], s2[1], u1, u2]/self.params.alpha)        
                policy[idx1 * len(actions2) + idx2] = prob
        # adjust for numerical errors in probability
        policy = policy / sum(policy)
        if verbose:
            print("ACTIONS1: ", actions1)
            print("ACTIONS2: ", actions1)
            print("state = ", s1, s2, " u = ", u1, u2)
            print("policy = ", policy)
        # return list of actions and list of probabilities
        return (actions1, actions2), policy
    
    def sample_trajectory(self,s1_0,s2_0, u1_0, u2_0, opt_value, max_t):
        xpositions1 = []
        xpositions2 = []
        ypositions1 = []
        ypositions2 = []
        u_states1 = []
        u_states2 = []
        
        xpositions1.append(s1_0[0])
        ypositions1.append(s1_0[1])
        xpositions2.append(s2_0[0])
        ypositions2.append(s2_0[1])
       
        u_states1.append(u1_0)
        u_states2.append(u2_0)
       
        s1 = s1_0.copy()
        s2 = s2_0.copy()
        u1 = u1_0
        u2 = u2_0
        
        value = np.zeros(max_t)
        for t in range(max_t):
            (actions1,actions2), policy = self.optimal_policy(s1, s2, u1, u2, opt_value, False)
            idx = np.random.choice(len(policy), p=policy)
            action1 = actions1[idx // len(actions2)]
            action2 = actions2[idx % len(actions2)]
            u1 = env.transition_u_one(s1,u1, self.arena)
            u2 = env.transition_u_one(s2,u2, self.arena)
            s1 = env.transition_s(s1, action1, self.arena)[0]
            s2 = env.transition_s(s2, action2, self.arena)[0]
            
            xpositions1.append(s1[0])
            ypositions1.append(s1[1])
            xpositions2.append(s2[0])
            ypositions2.append(s2[1])
            u_states1.append(u1)
            u_states2.append(u2)
            reward = -np.log(policy[idx])
            if t == 0:
                value[t] = reward
            else: 
                value[t] = value[t-1] + self.params.gamma**t * reward

        return xpositions1, ypositions1, u_states1, xpositions2, ypositions2, u_states2, value
    

########################################
###### Competitive MOP super-agent #####
########################################
class SuperAgentHCompet:
    def __init__(self, s01,s02, u, arena, params):
        self.s01 = s01
        self.s02 = s02
        self.u0 = u
        self.arena = arena
        self.params = params

    def h_iteration(self, tolerance, n_iter, verbose):
        value = np.zeros((self.arena.sizex, self.arena.sizey, self.arena.sizex, self.arena.sizey,self.arena.sizeu))
        value_old = deepcopy(value)
        t_stop = n_iter
        for t in range(n_iter):
            ferror_max = 0
            for u in range(self.arena.sizeu):
                for x1 in range(self.arena.sizex):
                    for y1 in range(self.arena.sizey):
                        for x2 in range(self.arena.sizex):
                            for y2 in range(self.arena.sizey):
                                s1 = [x1,y1]
                                s2 = [x2,y2]
                                actions1, _ = env.adm_actions(s1, u, self.arena, self.params)
                                actions2, _ = env.adm_actions(s2, u, self.arena, self.params)
                                Z = 0
                                for a1 in actions1:
                                    for a2 in actions2:
                                        s_primes1 = env.transition_s(s1, a1, self.arena)
                                        s_primes2 = env.transition_s(s2, a2, self.arena)
                                        expo = 0
                                        for s_prime1 in s_primes1:
                                            for s_prime2 in s_primes2:
                                                u_prime = env.transition_shared_u_av(s1,s2, u, self.arena)
                                                expo += self.params.gamma * value_old[s_prime1[0], s_prime1[1], s_prime2[0], s_prime2[1],u_prime]
                                        Z += np.exp(expo / self.params.alpha)   
                                value[x1,y1,x2,y2,u] = self.params.alpha * log(Z)
                                f_error = abs( value[x1, y1,x2,y2, u] - value_old[x1, y1,x2,y2,u])
                                ferror_max = max(ferror_max, f_error)
            if ferror_max < tolerance:
                t_stop = t
                break
            if verbose:
                print(f"iteration = {t}, max function error = {ferror_max}")
            value_old = deepcopy(value)
            with open('hvalue_compet.pkl', 'wb') as f:
                pickle.dump(value, f)
        return value, t_stop
    
    def optimal_policy(self, s1, s2, u, optimal_value, verbose):
        actions1, _ = env.adm_actions(s1, u, self.arena, self.params)
        actions2, _ = env.adm_actions(s2, u, self.arena, self.params)
        policy = np.zeros(len(actions1) * len(actions2))
        # Only compute policy for available actions
        for idx1, a1 in enumerate(actions1):
            for idx2, a2 in enumerate(actions2):
                u_p = env.transition_shared_u_av(s1, s2, u, self.arena)
                s_p1 = env.transition_s(s1, a1, self.arena)[0]
                s_p2 = env.transition_s(s2, a2, self.arena)[0]
                prob = np.exp(self.params.gamma*optimal_value[s_p1[0], s_p1[1], s_p2[0], s_p2[1], u_p]/self.params.alpha
                                            - optimal_value[s1[0], s1[1], s2[0], s2[1], u]/self.params.alpha)        
                policy[idx1 * len(actions2) + idx2] = prob
        # adjust for numerical errors in probability
        policy = policy / sum(policy)
        if verbose:
            print("ACTIONS1: ", actions1)
            print("ACTIONS2: ", actions1)
            print("state = ", s1, s2, " u = ", u)
            print("policy = ", policy)
        # return list of actions and list of probabilities
        return (actions1, actions2), policy
    
    def sample_trajectory(self,s1_0,s2_0, u_0, opt_value, max_t):
        xpositions1 = []
        xpositions2 = []
        ypositions1 = []
        ypositions2 = []
        u_states= []
    
        xpositions1.append(s1_0[0])
        ypositions1.append(s1_0[1])
        xpositions2.append(s2_0[0])
        ypositions2.append(s2_0[1])
       
        u_states.append(u_0)
    
        s1 = s1_0.copy()
        s2 = s2_0.copy()
        u = u_0
       
        value = np.zeros(max_t)
        for t in range(max_t):
            (actions1,actions2), policy = self.optimal_policy(s1, s2, u, opt_value, False)
            idx = np.random.choice(len(policy), p=policy)
            action1 = actions1[idx // len(actions2)]
            action2 = actions2[idx % len(actions2)]
            u = env.transition_shared_u_av(s1,s2, u, self.arena)
            s1 = env.transition_s(s1, action1, self.arena)[0]
            s2 = env.transition_s(s2, action2, self.arena)[0]
            
            xpositions1.append(s1[0])
            ypositions1.append(s1[1])
            xpositions2.append(s2[0])
            ypositions2.append(s2[1])
            u_states.append(u)
            reward = -np.log(policy[idx])
            if t == 0:
                value[t] = reward
            else: 
                value[t] = value[t-1] + self.params.gamma**t * reward
        return xpositions1, ypositions1, u_states, xpositions2, ypositions2, value
    
#######################################
###### Greedy super-agent class #######
#######################################
class SuperAgentGreedy:
    def __init__(self, s01,s02, u01,u02, arena, params,epsilon):
        self.s01 = s01
        self.s02 = s02
        self.u01 = u01
        self.u02 = u02
        self.arena = arena
        self.params = params
        self.epsilon = epsilon

    def val_iteration(self, tol, n_iter, verbose):
        value = np.zeros((self.arena.sizex, self.arena.sizey, self.arena.sizex, self.arena.sizey, self.arena.sizeu, self.arena.sizeu))
        epsilon = self.epsilon
        value_old = deepcopy(value)
        t_stop = n_iter
        for t in range(n_iter):
            ferror_max = 0
            for u1 in range(self.arena.sizeu):
                for u2 in range(self.arena.sizeu):
                    for x1 in range(self.arena.sizex):
                        for y1 in range(self.arena.sizey):
                            for x2 in range(self.arena.sizex):
                                for y2 in range(self.arena.sizey):
                                    s1 = [x1, y1]
                                    s2 = [x2, y2]
                                    actions1, _ = env.adm_actions(s1, u1, self.arena, self.params)
                                    actions2, _ = env.adm_actions(s2, u2, self.arena, self.params)
                                    values = np.zeros((len(actions1), len(actions2)))
                                    for id_a1, a1 in enumerate(actions1):
                                        for id_a2, a2 in enumerate(actions2):
                                            s_primes1 = env.transition_s(s1, a1, self.arena)
                                            s_primes2 = env.transition_s(s2, a2, self.arena)
                                            r1 = env.reachable_food(s1, u1, self.arena)
                                            r2 = env.reachable_food(s2, u2, self.arena)
                                            for s_prime1 in s_primes1:
                                                for s_prime2 in s_primes2:
                                                    u_prime1 = env.transition_u(s1, s2, u1, self.arena)
                                                    u_prime2 = env.transition_u(s2, s1, u2, self.arena)
                                                    values[id_a1, id_a2] += r1 + r2 + self.params.gamma * value_old[s_prime1[0], s_prime1[1], s_prime2[0], s_prime2[1], u_prime1, u_prime2]
                                    value[x1, y1, x2, y2, u1, u2] = (1 - epsilon) * np.max(values) + (epsilon / (len(actions1) * len(actions2))) * np.sum(values)
                                    f_error = abs(value[x1, y1, x2, y2, u1, u2] - value_old[x1, y1, x2, y2, u1, u2])
                                    ferror_max = max(ferror_max, f_error)
            if ferror_max < tol:
                t_stop = t
                break
            if verbose:
                print(f"iteration = {t}, max function error = {ferror_max}")
            value_old = deepcopy(value)
        return value, t_stop
    
    def optimal_policy_q(self, s1, s2, u1, u2, optimal_value, verbose):
        epsilon = self.epsilon
        actions1, _ = env.adm_actions(s1, u1, self.arena, self.params)
        actions2, _ = env.adm_actions(s2, u2, self.arena, self.params)
        q_values = np.zeros(len(actions1) * len(actions2))
        policy = np.zeros(len(actions1) * len(actions2))
        # only compute policy for available actions
        for idx1, a1 in enumerate(actions1):
            for idx2, a2 in enumerate(actions2):
                r1 = env.reachable_food(s1, u1, self.arena)
                r2 = env.reachable_food(s2, u2, self.arena)
                u_p1 = env.transition_u(s1, s2, u1, self.arena)
                u_p2 = env.transition_u(s2, s1, u2, self.arena)
                s_p1 = env.transition_s(s1, a1, self.arena)[0]
                s_p2 = env.transition_s(s2, a2, self.arena)[0]
                q_values[idx1 * len(actions2) + idx2] += r1 + r2 + self.params.gamma * optimal_value[s_p1[0], s_p1[1], s_p2[0], s_p2[1], u_p1, u_p2]
        best_actions = [i for i, x in enumerate(q_values) if x == max(q_values)]
        for i in range(len(actions1)):
            for j in range(len(actions2)):
                if i in best_actions and j in best_actions:
                    policy[i * len(actions2) + j] = (1 - epsilon) / len(best_actions) + epsilon / (len(actions1) * len(actions2))
                else: 
                    policy[i*len(actions2) + j] = epsilon / (len(actions1) * len(actions2))
        # adjust for numerical errors in probability
        policy = policy / sum(policy)
        if verbose:
            print("state = ", s1, s2, " u = ", u1, u2)
            print("policy = ", policy)
        # return list of actions and list of probabilities
        return (actions1, actions2), policy
    
    def sample_trajectory(self,s1_0,s2_0, u1_0, u2_0, opt_value, max_t):
        xpositions1 = []
        xpositions2 = []
        ypositions1 = []
        ypositions2 = []
        u_states1 = []
        u_states2 = []
        
        xpositions1.append(s1_0[0])
        ypositions1.append(s1_0[1])
        xpositions2.append(s2_0[0])
        ypositions2.append(s2_0[1])
       
        u_states1.append(u1_0)
        u_states2.append(u2_0)
        
        s1 = s1_0.copy()
        s2 = s2_0.copy()
        u1 = u1_0
        u2 = u2_0
        
        value = np.zeros(max_t)
        for t in range(max_t):
            (actions1,actions2), policy = self.optimal_policy_q(s1, s2, u1, u2, opt_value, False)
            idx = np.random.choice(len(policy), p=policy)
            action1 = actions1[idx // len(actions2)]
            action2 = actions2[idx % len(actions2)]

            u1 = env.transition_u(s1,s2, u1, self.arena)
            u2 = env.transition_u(s2,s1, u2, self.arena)
            s1 = env.transition_s(s1, action1, self.arena)[0]
            s2 = env.transition_s(s2, action2, self.arena)[0]
            
            xpositions1.append(s1[0])
            ypositions1.append(s1[1])
            xpositions2.append(s2[0])
            ypositions2.append(s2[1])
            u_states1.append(u1)
            u_states2.append(u2)
            reward = env.reachable_food(s1, u1, self.arena) + env.reachable_food(s2, u2, self.arena)
            if t == 0:
                value[t] = reward
            if t > 0: 
                value[t] = value[t-1] + self.params.gamma**t*reward
        return xpositions1, ypositions1, u_states1, xpositions2, ypositions2, u_states2, value

##########################
##### Random walk ########
########################## 
def sample_trajectory_random(s_0, s_1, u_0, u_1, max_t, arena, params):
    xpositions_0, ypositions_0, u_states_0 = [], [], []
    xpositions_1, ypositions_1, u_states_1 = [], [], []
    xpositions_0.append(s_0[0])
    ypositions_0.append(s_0[1])
    u_states_0.append(u_0)
    xpositions_1.append(s_1[0])
    ypositions_1.append(s_1[1])
    u_states_1.append(u_1)
    
    s_0, s_1 = deepcopy(s_0), deepcopy(s_1)
    u_0, u_1 = deepcopy(u_0), deepcopy(u_1)
   
    for t in range(max_t):
        actions_at_s_0, _ = env.adm_actions(s_0, u_0, arena, params)
        actions_at_s_1, _ = env.adm_actions(s_1, u_1, arena, params)
        
        idx_0 = np.random.choice(len(actions_at_s_0))
        idx_1 = np.random.choice(len(actions_at_s_1))
        
        action_0 = actions_at_s_0[idx_0]
        action_1 = actions_at_s_1[idx_1]
        
        u_0 = env.transition_u_one(s_0, u_0, arena)
        u_1 = env.transition_u_one(s_1, u_1, arena)
        
        s_0 = env.transition_s(s_0, action_0, arena)[0]
        s_1 = env.transition_s(s_1, action_1, arena)[0]
        
        xpositions_0.append(s_0[0])
        ypositions_0.append(s_0[1])
        u_states_0.append(u_0)
        
        xpositions_1.append(s_1[0])
        ypositions_1.append(s_1[1])
        u_states_1.append(u_1)
        
    return xpositions_0, ypositions_0, u_states_0, xpositions_1, ypositions_1, u_states_1