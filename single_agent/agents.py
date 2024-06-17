import environment as env
from copy import deepcopy
from math import exp, log
import numpy as np


class MOPAgent:
    def __init__(self, s, u, arena, params):
        self.s = s
        self.u = u
        self.arena = arena
        self.params = params

    def h_iteration(self, tolerance, n_iter, verbose):
        value = np.zeros((self.arena.sizex, self.arena.sizey, self.arena.sizeu))
        value_old = deepcopy(value)
        t_stop = n_iter
        f_error = 0
        for t in range(n_iter):
            ferror_max = 0
            for u in range(self.arena.sizeu):
                for x in range(self.arena.sizex):
                    for y in range(self.arena.sizey):
                        s = [x, y]
                        actions, _ = env.adm_actions(s, u, self.arena, self.params)
                        Z = 0
                        for a in actions:
                            s_primes = env.transition_s(s, a, self.arena)
                            expo = 0
                            for s_prime in s_primes:
                                u_prime = env.transition_u_one(s, u, self.arena)
                                expo += self.params.gamma * value_old[s_prime[0], s_prime[1], u_prime]
                            Z += exp(expo / self.params.alpha)
                        value[x, y, u] = self.params.alpha * log(Z)
                        f_error = abs(value[x, y, u] - value_old[x, y, u])
                        ferror_max = max(ferror_max, f_error)
                        
                
                #print(f"VALUE AT ITER {t} = {value}")

            if ferror_max < tolerance:
                t_stop = t
                break
            if verbose:
                print(f"iteration = {t}, max function error = {ferror_max}")
            value_old = deepcopy(value)
        return value, t_stop

    def optimal_policy(self, s, u, optimal_value, verbose):
        actions, _ = env.adm_actions(s, u, self.arena, self.params)
        policy = [0]*len(actions)
        # Only compute policy for available actions
        for idx, a in enumerate(actions):
            u_p = env.transition_u_one(s, u, self.arena)
            s_p = env.transition_s(s, a, self.arena)[0]
            policy[idx] = np.exp(self.params.gamma*optimal_value[s_p[0], s_p[1], u_p]/self.params.alpha - optimal_value[s[0], s[1], u]/self.params.alpha)
        # adjust for numerical errors in probability
        sum_p = sum(policy)
        if verbose:
            print("state = ", s, " u = ", u)
            print("policy = ", policy)
            print("sum policy = ", sum_p)
        policy = [p/sum_p for p in policy]
       
        # return list of actions and list of probabilities
        return actions, policy
    
    def sample_trajectory(self, s, u, opt_value, max_t):
        xpositions = []
        ypositions = []
        u_states = []
        
        xpositions.append(s[0])
        ypositions.append(s[1])
        u_states.append(u)

        s = s.copy()
        u = u
        
        value = np.zeros(max_t)
        for t in range(max_t):
            actions, policy = self.optimal_policy(s, u, opt_value, False)
            idx = np.random.choice(len(policy), p=policy)
            action = actions[idx]
          
            u = env.transition_u_one(s, u, self.arena)
            s = env.transition_s(s, action, self.arena)[0]  

            xpositions.append(s[0])
            ypositions.append(s[1])
            u_states.append(u)
            reward = -np.log(policy[idx])
            if t == 0:
                value[t] = reward
            else: 
                value[t] = value[t-1] + self.params.gamma**t * reward

        return xpositions, ypositions, u_states, value

    
class GreedyAgent:
    def __init__(self, s, arena, params):
        self.s = s
        #self.u0 = u0
        self.arena = arena
        self.params = params
    
    def q_iteration(self, epsilon,tol, n_iter, verbose):
        value = np.zeros((self.arena.sizex, self.arena.sizey))
        value_old = deepcopy(value)
        t_stop = n_iter
        f_error = 0

        for t in range(n_iter):
            ferror_max = 0
            for x in range(self.arena.sizex):
                for y in range(self.arena.sizey):
                    s = [x, y]
                    actions, _ = env.adm_actions(s, 1,self.arena, self.params)
                    values = np.zeros(len(actions))
                    for id_a, a in enumerate(actions):
                        s_primes =  env.transition_s(s, a, self.arena)
                        r = env.reachable_food(s, self.arena)
                        for s_prime in s_primes:
                            values[id_a] += r + self.params.gamma*value_old[s_prime[0],s_prime[1]]
                    value[x,y] = (1-epsilon)*np.max(values) + (epsilon/len(actions))*np.sum(values)
                    f_error = abs(value[x,y] - value_old[x,y])
                    ferror_max = max(ferror_max,f_error)
            if ferror_max < tol:
                t_stop = t
                break
            if verbose: 
                print("iteration = ", t, ", max function error = ", ferror_max)
            value_old = deepcopy(value)
        return value, t_stop
    

    def optimal_policy_q(self,s, value, epsilon):
        actions, ids_actions = env.adm_actions(s, 1, self.arena, self.params)
        q_values = [0]*len(actions)
        policy = [0]*len(actions)
        for idx, a in enumerate(actions):
            r = env.reachable_food(s, self.arena)
            s_p = env.transition_s(s, a, self.arena)[0]
            q_values[idx] += r + self.params.gamma*value[s_p[0], s_p[1]]
        best_actions = [i for i, x in enumerate(q_values) if x == max(q_values)]
        for i in range(len(actions)):
            if i in best_actions:
                policy[i] = (1-epsilon)/len(best_actions) + epsilon/len(actions)
            else:
                policy[i] = epsilon/len(actions)
        return actions, policy
    
    def sample_trajectory_q(self, s, opt_value, max_t,epsilon):
        xpositions = []
        ypositions = []
        
        xpositions.append(s[0])
        ypositions.append(s[1])
    
        s = s.copy()
        value = np.zeros(max_t)
        for t in range(max_t):
            actions, policy = self.optimal_policy_q(s, opt_value, epsilon)
            idx = np.random.choice(len(policy), p=policy)
            action = actions[idx]
            s = env.transition_s(s, action, self.arena)[0]

            xpositions.append(s[0])
            ypositions.append(s[1])
            reward = env.reachable_food(s, self.arena)
            if t == 0:
                value[t] = reward
            else: 
                value[t] = value[t-1] + self.params.gamma**t * reward
        return xpositions, ypositions, value
