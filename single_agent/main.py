import numpy as np
import environment as env
import agents as ag
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import animations as animate

params = env.Paras(alpha=1, constant_actions=False)

# define and initialize environment
size_x = 20
size_y = 5
capacity = 50
food_gain = 5
reward_locs = [[19,2]]
reward_mags = [food_gain]
arena = env.initialize_arena(size_x,size_y, capacity+1,reward_locs, reward_mags)

# initial conditions
s_0 = [2,2] 
u_0 = capacity #17#10

mop_agent = ag.MOPAgent(s_0, u_0, arena, params)
#greedy_agent = ag.GreedyAgent(s_0, arena, params)

env.draw_arena(1, [mop_agent.s], [mop_agent.u], arena, params)
#env.draw_arena(1, [greedy_agent.s], [greedy_agent.u0], arena, params)

verbose = True
tol= 0.01
n_iter =300
epsilon = 0.4
comp=False
if comp:
    print("computing....")
    hvalue, t_stop= mop_agent.h_iteration(tol, n_iter, verbose)
    #qvalue, t_stop = greedy_agent.q_iteration(epsilon, tol, n_iter, verbose)
    with open('mop_20x5_onefshighg.pkl', 'wb') as f:
       pickle.dump((hvalue, t_stop), f)
else:
    with open('mop_20x5_onefshighg.pkl', 'rb') as f:
        hvalue, t_stop = pickle.load(f)

##############################
# simulate single trajectory #
##############################
single_sim = False
if single_sim:
    #xpositions,ypositions,_ = greedy_agent.sample_trajectory_q(s_0, qvalue, 250,0)
    xpositions,ypositions,u_states,value = mop_agent.sample_trajectory(s_0, u_0, hvalue, 250)

    #simulate_greedy(arena, xpositions, ypositions, 'single_trajGreedy')
    animate.simulate(arena, xpositions, ypositions, u_states, value, 'single_trajMOP')

##############################
### avg value vs. optimal ####
##############################
num_trajectories=1000
# List to store the values
values = []
xpos =[]
ypos =[]
max_t = 250
avg_vs_plot = False
if avg_vs_plot:
    for _ in range(num_trajectories):
        #_,_,_, value = greedy_agent.sample_trajectory_q(s_0, u_0, qvalue, max_t,epsilon)
        xpositions,ypositions,ustates, value = mop_agent.sample_trajectory(s_0,u_0, hvalue, max_t)
        values.append(value)
        xpos.append(xpositions)
        ypos.append(ypositions)

    with open('moptrajectories.pkl', 'wb') as f:
        pickle.dump((values,xpos, ypos), f)
    avg_traj = np.mean(values, axis=0)

    time = list(range(len(values[0])))
    values = np.array(values)
    hval =  hvalue[s_0[0],s_0[1], u_0].T
    # Plot each trajectory
    for i in range(num_trajectories):
        plt.plot(time, values[i], alpha=0.5, color='grey', label='Return over single trajectory' if i == 0 else None)

    # Plot the average trajectory
    plt.plot(time, avg_traj, color='red', linewidth=2, label = 'Average return')
    plt.plot(time, np.full_like(values[0], hval), color='black', linewidth=2,linestyle='--', label='Expected return')
   

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(np.arange(min(time), max(time)+1,50))
    plt.legend(loc='center right')
    plt.show()

###############################
### heatmap of trajectories ###
###############################
anim = True
if anim:
    num_trajectories = 1000

    # Run the function in a loop
    for _ in range(num_trajectories):
        #_,_,_, value = ag1.sample_trajectory_q(s_0, u_0, hvalue, max_t)
        xpositions,ypositions,_,_ = mop_agent.sample_trajectory(s_0, u_0,hvalue, max_t)
        #xpositions,ypositions,_ = ag.sample_trajectory_random(s_0, u_0, max_t,arena,params)
        #xpositions,ypositions,_, value = greedy_agent.sample_trajectory_q(s_0, u_0, qvalue, max_t,epsilon)
        xpos.append(xpositions)
        ypos.append(ypositions)

    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((20, 5)), origin='lower', cmap="hot", interpolation='nearest',vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)

    def update(frame):
        # Clear the scatter plots only
        ax.collections.clear()

        ax.set_xlim(0, 20)  
        ax.set_ylim(0, 5)  
        heatmap, xedges, yedges = np.histogram2d(
            [x[frame] + 0.5 for x in xpos],
            [y[frame] + 0.5 for y in ypos],
            bins=[np.arange(0, 21), np.arange(0, 6)],
            density=True
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im.set_data(heatmap.T)
        im.set_extent(extent)
        im.set_clim(0, 1)  
        for i in range(len(arena.reward_locs)):
            ax.scatter([arena.reward_locs[i][0] + 0.5], [arena.reward_locs[i][1] + 0.5], s=arena.reward_mags[i]*3, color='green', marker='D')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        return im,
    ani = animation.FuncAnimation(fig, update, frames=len(xpos[0]), interval=200)

    ani.save('heatmapMOPjust'+'.gif', writer='pillow')
    plt.show()



###################################################
#### plot optimal policy and heatmap of values ####
###################################################
def plot_optimal_policy(plot, u, opt_value, arena, arrow_length=1.5, arrow_width=0.02):
    for x in range(arena.sizex):
        for y in range(arena.sizey):
            actions, probs = mop_agent.optimal_policy([x,y], u, opt_value, False)
            
            arrow_x = np.zeros(len(actions))
            arrow_y = np.zeros(len(actions))
            aux = np.array(actions) * np.array(probs)[:, None]
            for i in range(len(aux)):
                arrow_x[i] = 5*aux[i][0] * arrow_length                  
                arrow_y[i] =  5*aux[i][1] * arrow_length    
            for i in range(len(aux)):
                arrow = patches.FancyArrowPatch((x, y), (x + arrow_x[i], y + arrow_y[i]), arrowstyle='->', color='k', mutation_scale=10)
                plot.add_patch(arrow)
            # plot dot proportional to the prob of doing nothing (a=[0,0])
            plot.scatter(x, y, s=probs[-1] * 100, color='red')
            
plot_optpolicy = False
if plot_optpolicy:
    fig, ax = plt.subplots()
    cax = ax.imshow(hvalue[:, :, u_0].T, origin='lower', cmap='viridis')
    cbar = fig.colorbar(cax)
    plot_optimal_policy(ax, u_0, hvalue, arena, arrow_length=0.3, arrow_width=1)
    ax.axis('off')
    plt.show()