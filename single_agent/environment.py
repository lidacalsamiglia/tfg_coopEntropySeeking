import matplotlib.pyplot as plt
import numpy as np
import copy
from operator import add

##############################
##  environment definition  ##
##############################

class Arena:
    def __init__(self, sizex, sizey, sizeu, xborders=None, yborders=None, 
				foodsource_locs=None, foodsource_mags=None):
        self.sizex = sizex
        self.sizey = sizey
        self.sizeu = sizeu
        self.xborders = xborders if xborders is not None else [-1, sizex]
        self.yborders = yborders if yborders is not None else [-1, sizey]
        self.foodsource_locs = foodsource_locs
        self.foodsource_mags = foodsource_mags

class Paras:
    def __init__(self, gamma=0.98, alpha=1, beta=0, constant_actions=True):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.constant_actions = constant_actions

def adm_actions(s_state, u_state, arena, pars):
    out = []
    moving_actions = [[0,1],[0,-1],[1,0],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]
    ids_actions = list(range(1, len(moving_actions) + 1))
    # If agent is "alive"
    if u_state > 0:
        out = copy.deepcopy(moving_actions)
        # When we constrain by hand the amount of actions
        # we delete some possible actions
        if not pars.constant_actions:
            # Give all possible actions by default
            # Check the boundaries of gridworld
            for it in range(2):
                not_admissible = s_state[0] + (-1)**it in arena.xborders
                if not_admissible:
                    ids = [i for i, idx in enumerate(out) if idx[0] == (-1)**it]
                    for i in sorted(ids, reverse=True):
                        del out[i]
                        del ids_actions[i]
            for it in range(2):
                not_admissible = s_state[1] + (-1)**it in arena.yborders
                if not_admissible:
                    ids = [i for i, idx in enumerate(out) if idx[1] == (-1)**it]
                    for i in sorted(ids, reverse=True):
                        del out[i]
                        del ids_actions[i]
    else:
        ids_actions = []
    # Doing nothing is always an admissible action
    out.append([0,0])
    ids_actions.append(len(ids_actions) + 1)
    return out, ids_actions


def initialize_arena(size_x, size_y, capacity, foodsource_locs, foodsource_mags):
    return Arena(sizex=size_x, sizey=size_y, sizeu=capacity, foodsource_locs=foodsource_locs, foodsource_mags=foodsource_mags)


def draw_arena(num_agents, initial_positions, initial_states, arena, pars):
    fig, ax = plt.subplots()
    for i in range(len(arena.foodsource_locs)):
        ax.scatter([arena.foodsource_locs[i][0] + 0.5], [arena.foodsource_locs[i][1] + 0.5], s=arena.foodsource_mags[i]*3, color='green', marker='D')
    ax.set_xticks(np.arange(0, arena.sizex + 1, 1))
    ax.set_yticks(np.arange(0, arena.sizey + 1, 1))
    ax.grid(True)
    ax.set_xlim([0, arena.sizex])
    ax.set_ylim([0, arena.sizey])

    # Set the aspect ratio of the plot to be equal
    ax.set_aspect('equal')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)


    for i in range(num_agents):
        x_pos, y_pos = initial_positions[i]
        u = initial_states[i]
        # Draw arrows
        actions = adm_actions([x_pos, y_pos], u, arena, pars)[0]
        arrow_x = [action[0]*2.2/np.sqrt(2) if np.linalg.norm(action) > 1 else action[0]*2.2 for action in actions]
        arrow_y = [action[1]*2.2/np.sqrt(2) if np.linalg.norm(action) > 1 else action[1]*2.2 for action in actions]
        ax.quiver([x_pos + 0.5]*len(actions), [y_pos + 0.5]*len(actions), arrow_x, arrow_y, color='black', linewidth=0.2)
        # Draw agent
        ax.scatter(x_pos + 0.5, y_pos + 0.5, s=15, color='orange', marker='^')
    plt.show()
    
def transition_s(s, a, arena):
    s_prime = [sum(x) for x in zip(s, a)]
    if s_prime[0] == arena.xborders[0] or s_prime[0] == arena.xborders[1] or s_prime[1] == arena.yborders[0] or s_prime[1] == arena.yborders[1]:
        s_prime = s    
    return [s_prime]

    
def food_one(s, arena):
    food = 0
    for i in range(len(arena.foodsource_locs)):
        if s == arena.foodsource_locs[i]:
            food += arena.foodsource_mags[i]
    # substract one for each time step
    return food -  1

def transition_u_one(s,u,arena):
    u_prime = u + food_one(s,arena)
    if u_prime >= arena.sizeu:
        u_prime = arena.sizeu-1
    elif u_prime < 0:
        u_prime = 0
    return u_prime
        

## Greedy
def reachable_food(s, arena, delta_reward=1):
    r = 0
    for i in range(len(arena.foodsource_locs)):
        if s == arena.foodsource_locs[i]:
            r = delta_reward
    return r