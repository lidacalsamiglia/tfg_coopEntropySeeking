import numpy as np
import environment as env
import agents as ag
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import pickle
import animations_plots as animate 


params = env.Paras(alpha=1, constant_actions=False)

# Define and initialize environment
size_x = 11#5
size_y = 5
capacity = 25#10
food_gain = 5
foodsource_locs = [[0,2],[10,2]]
foodsource_mags = [food_gain,2*food_gain]
arena = env.initialize_arena(size_x,size_y, capacity+1,foodsource_locs, foodsource_mags)

# Initial condition
s1_0 = [2,1] 
u_0 = capacity
s2_0 = [2,3]
u_1 = capacity

# Super agent initialization
super_agent = ag.SuperAgentHCollab(s1_0, s2_0,u_0,u_1, arena, params)
#super_agent = ag.SuperAgentHNonCollab(s1_0, s2_0,u_0,u_1, arena, params)
#super_agent = ag.SuperAgentHCompet(s1_0, s2_0, u_0, arena, params)

# Draw arena
#env.draw_arena(2, [super_agent.s01, super_agent.s02], [super_agent.u01, super_agent.u02], arena, params)

verbose = True
tol= 0.01
n_iter = 600

load = True
if load:
    comp=False
    if comp:
        h_value, t_stop_indep= super_agent.h_iteration(tol, n_iter, verbose)
        # save to a file
        with open('final_hbothin.pkl', 'wb') as f:
            pickle.dump((h_value, t_stop_indep), f)
    else:
        # load to a file
        with open('collab/final_hbothin.pkl', 'rb') as f:
            h_value, t_stop_indep = pickle.load(f)



##########################################
###### Value function visualization ######
##########################################
show = False
if show:
    sc = [1,4]
    h1 =  h_value[:, :, s2_0[0], s2_0[1],u_0, u_1].T
    h2 = h_value[:, :, sc[0], sc[1], u_0, 11].T
    #h1 =  h_value[:, :, s2_0[0], s2_0[1],u_0].T
    #h2 = h_value[:, :, sc[0], sc[1], u_0].T
    h = h1 - h2
    print("hval: ",h)
    plt.figure(figsize=(10, 8))
    # Adjusting x and y limits to match draw environment function
    h_rounded = np.round(h, 2)
    sns.heatmap(h1, annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks([]) 
    plt.yticks([]) 
    plt.gca().invert_yaxis()
    plt.show()
    plt.figure(figsize=(10, 8))
    sns.heatmap(h2, annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks([]) 
    plt.yticks([]) 
    plt.gca().invert_yaxis()
    plt.show()
    plt.figure(figsize=(10, 8))
    sns.heatmap(h_rounded, annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks([]) 
    plt.yticks([]) 
    plt.gca().invert_yaxis()
    plt.show()

####################################
###### Avg value vs. optimal #######
####################################
num_trajectories = 1000  
max_t = 250 
plot = False
if plot:
    values = []
    xpos1 = []
    ypos1 = []
    xpos2 = []
    ypos2 = []

    for _ in range(num_trajectories):
        #_,_,_, value = greedy_agent.sample_trajectory_q(s_0, u_0, qvalue, max_t,epsilon)
        xpositions1, ypositions1, _, xpositions2, ypositions2, _, value = super_agent.sample_trajectory(s1_0,s2_0, u_0, u_1, h_value, max_t)
        #xpositions1, ypositions1, u_states, xpositions2, ypositions2, value = super_agent.sample_trajectory(s1_0,s2_0, u_0, h_value, max_t)
        values.append(value)
        xpos1.append(xpositions1)
        ypos1.append(ypositions1)
        xpos2.append(xpositions2)
        ypos2.append(ypositions2)


    avg_traj = np.mean(values, axis=0)

    time = list(range(max_t))
    values = np.array(values)
    hval = h_value[s1_0[0], s1_0[1], s2_0[0], s2_0[1],u_0, u_1].T

    #hval = h_value[s1_0[0], s1_0[1], s2_0[0], s2_0[1],u_0].T
    # plot each trajectory
    for i in range(num_trajectories):
        plt.plot(time, values[i], alpha=0.5, color='grey', label='Return over single trajectory' if i == 0 else None)

    # plot the average trajectory
    plt.plot(time, avg_traj, color='red', linewidth=2, label = 'Average return')
    plt.plot(time, np.full_like(values[0], hval), color='black', linewidth=2,linestyle='--', label='Expected return')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks(np.arange(min(time), max(time)+1,50))
    plt.legend(loc='center right')
    plt.show()

anim = False
if anim:
    xpositions1, ypositions1, u_states1, xpositions2, ypositions2, u_states2, value = super_agent.sample_trajectory(s1_0,s2_0, u_0, h_value, max_t)
    with open('singletrajCOLLAB.pkl', 'wb') as f:
        pickle.dump((xpositions1, ypositions1, u_states1, xpositions2, ypositions2, u_states2, value), f)

    '''with open('singletrajCOLLAB.pkl', 'rb') as f:
        xpositions1, ypositions1, u_states1, xpositions2, ypositions2, u_states2, value = pickle.load(f)'''
    animate.simulate(arena,xpositions1, ypositions1, u_states1, xpositions2, ypositions2, u_states2,value, "single_collab")
    #animate.simulate_share(arena, xpositions1, ypositions1, xpositions2, ypositions2, u_states, value, "SINGLESHARED")


#################################
###### Check independence #######
#################################
hdiff =[]
independent = True
check = True
if check:
    try:
        for u0 in range(capacity+1):
            for x in range(size_x):
                for y in range(size_y):
                    for x1 in range(size_x):
                        for y1 in range(size_y):
                            for u1 in range(capacity+1):
                                for u2 in range(capacity+1):
                                    h1 = h_value[:, :, x, y, u0, u1].T
                                    h2 = h_value[:, :, x1, y1, u0, u2].T
                                    d = h1-h2
                                    hdiff.append(d)
                                    if not np.all(np.round((d-d[0,0]),2) == 0):
                                        print("mat: ",d)
                                        print("first", d[0, 0])
                                        independent = False
                                        raise StopIteration 
    except StopIteration:
        pass
    print("independent: ", independent)
check_share= False
if check_share:
    try:
        for u0 in range(capacity+1):
            for x in range(size_x):
                for y in range(size_y):
                    for x1 in range(size_x):
                        for y1 in range(size_y):
                            h1 = h_value[:, :, x, y, u0].T
                            h2 = h_value[:, :, x1, y1, u0].T
                            d = h1-h2
                            hdiff.append(d)                                
                            if not np.all(np.round((d-d[0,0]),2) == 0):
                                print("mat: ",d)
                                print("first", d[0, 0])
                                independent = False
                                raise StopIteration  
    except StopIteration:
        pass
    print("INDEPENDENT: ", independent)
# equivalent way of cheking independence
'''s2s = [1,4]
s2ss = [4,4]
v1 = h_value[:,:,s2s[0],s2s[1],:,3].T 
v2 = h_value[s2ss[0],s2ss[1],:,:,3,:].T - v1[s2ss[0],s2ss[1],3]
indep = False
for u0 in range(capacity+1):
        for x in range(size_x):
            for y in range(size_y):
                for x1 in range(size_x):
                    for y1 in range(size_y):
                        for u1 in range(capacity+1):
                            for u2 in range(capacity+1):
                                if h_value[x,y,x1,y1,u0,u1]-(v1[x,y,u0]+v2[x1,y1,u1])==0:
                                    indep=True

print("INDEP: ",indep)'''

##################################################
##### Measure correlation btw trajectories #######
##################################################

# Mutual Information # 
mi = False
if mi:
    max_t = 250
    num_trajectories = 500000
    xpos1  =[]
    ypos1= []
    u_states1=[]
    xpos2  =[]
    ypos2= []
    u_states2=[]

    p1 = np.zeros((max_t, size_x, size_y))
    p2 = np.zeros((max_t, size_x, size_y))
    pjoint = np.zeros((max_t, size_x, size_y, size_x, size_y))

    mutual_info = np.zeros((num_trajectories, max_t))

    for i in range(num_trajectories):
        xpositions1, ypositions1,ustates1, xpositions2, ypositions2, ustates2,value = super_agent.sample_trajectory(s1_0,s2_0, u_0, u_1, h_value, max_t)
        xpos1.append(xpositions1)
        ypos1.append(ypositions1)
        #u_states1.append(ustates1)
        xpos2.append(xpositions2)
        ypos2.append(ypositions2)
        #u_states2.append(ustates2)
        for t in range(max_t):
            p1[t, xpos1[i][t], ypos1[i][t]] += 1
            p2[t, xpos2[i][t], ypos2[i][t]] += 1
            pjoint[t, xpos1[i][t], ypos1[i][t], xpos2[i][t], ypos2[i][t]] += 1

        # normalize probabilities
        p1_norm = p1 / (i+1)
        p2_norm = p2 / (i+1)
        pjoint_norm = pjoint / (i+1)
        p1_norm =  np.where(p1_norm==0,1e-100,p1_norm)
        p2_norm = np.where(p2_norm==0, 1e-100, p2_norm)
        pjoint_norm = np.where(pjoint_norm==0, 1e-100, pjoint_norm)

        for t in range(max_t):  
            entropy1 = -np.sum(p1_norm[t] * np.log(p1_norm[t]))
            entropy2 = -np.sum(p2_norm[t] * np.log(p2_norm[t]))
            joint_entropy = -np.sum(pjoint_norm[t] * np.log(pjoint_norm[t]))
            
            # compute mutual information for the current time step
            mutual_info[i, t] = entropy1 + entropy2 - joint_entropy
        
        # save mutual information every 100k trajectories
        if (i + 1) % 100000 == 0:
            with open(f'mutual_info_{i}collabpkl', 'wb') as f:
                pickle.dump(mutual_info, f)

    # save trajectories
    with open('trajectories_collab.pkl', 'wb') as f:
        pickle.dump((xpos1, ypos1, xpos2, ypos2, ustates1, ustates2), f)

create_heatmap = True
if create_heatmap:
    max_t = 250
    with open('trajectories_collab.pkl', 'rb') as f:
        xpos1, ypos1, xpos2, ypos2, ustates,_ = pickle.load(f)
    with open('mutual_info_collab.pkl', 'rb') as f:
        mutual_info_temp = pickle.load(f)
        mutual_info = mutual_info_temp[:499999, :]
    mi_av = mutual_info[499999 - 1]
    time = list(range(250))

    animate.heatmap_MI(xpos1, ypos1, xpos2, ypos2, mi_av, time, "MI_compet")

'''with open('noncollab/mutual_info_INDEP.pkl', 'rb') as f:
        mutual_info_temp = pickle.load(f)
        mutual_info = mutual_info_temp[:499999, :]
with open('mutual_info_RANDOM_final.pkl', 'rb') as f:
    mutual_info_temp = pickle.load(f)
    mutual_info_random = mutual_info_temp[:499999, :]
mi_av = mutual_info[499999 - 1]
mi_random = mutual_info_random[499999 - 1]
time = list(range(250))
plt.plot(time, mi_av, label='MI Non-collaborative', color ='red')
plt.plot(time, mi_random, label='MI Random',color='cyan',alpha=0.5)
plt.legend(loc='upper right')
plt.ylabel('Mutual Information')
plt.xlabel('Time')
plt.ylim(0,0.01)
plt.show()'''

# Covariance Matrix #
compute_covs = False
if compute_covs:
    with open('trajectories_collab.pkl', 'rb') as f:
        xpos1, ypos1, xpos2, ypos2, _, _ = pickle.load(f)
    max_t = 250
    covs = []
    for t in range(max_t):
        locs_t = [[xpos1[i][t], ypos1[i][t], xpos2[i][t], ypos2[i][t]] for i in range(len(xpos1))]
        cov_t = np.cov(np.array(locs_t).T)
        covs.append(cov_t)

    avg_cov = np.mean(covs, axis=0)
    avg_cov_abs= np.mean(np.abs(covs), axis=0)

    fig = plt.figure()
    def update(i):
        plt.clf()
        sns.heatmap(covs[i], annot=True, cmap='coolwarm')
    plt.xticks([])  
    plt.yticks([])  
    
    ani = animation.FuncAnimation(fig, update, frames=range(max_t), repeat=False)
    ani.save('covsINDEP.gif', writer='pillow')
