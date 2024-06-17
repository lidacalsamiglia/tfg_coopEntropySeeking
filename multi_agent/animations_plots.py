import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def simulate(arena, xpositions1, ypositions1, u_states1, xpositions2, ypositions2, u_states2, values, filename):
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    ax_value = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])
    ax.set_aspect('equal')

    # initialize agents and energy plots
    leg1_plot, = ax.plot([], [], '^', color='orange', markersize=10)
    leg2_plot, = ax.plot([], [], '^', color='blue', markersize=10)
    energy_bar1 = ax_energy.barh([1], [0], color='orange', height=0.5)
    energy_bar2 = ax_energy.barh([2], [0], color='blue', height=0.5)
    ax_energy.set_xlim(0, max(u_states1 + u_states2) * 1.2)  # added extra space on the side
    ax_energy.set_ylim(0, 3)
    ax_energy.set_yticks([1, 2])
    ax_energy.set_yticklabels(['Agent 1', 'Agent 2'])
    ax_energy.set_xlabel('Energy')

    # initialize a plot element for the value
    value_plot, = ax_value.plot([], [], color='blue')
    ax_value.set_xlim(0, len(values))
    ax_value.set_ylim(min(values), max(values))
    ax_value.set_xlabel('Time')
    ax_value.set_ylabel('Value')

    def init():
        ax.set_xlim(0, arena.sizex)
        ax.set_ylim(0, arena.sizey)
        ax.set_xticks(np.arange(0, arena.sizex + 1, 1))
        ax.set_yticks(np.arange(0, arena.sizey + 1, 1))
        ax.grid(True)
        return leg1_plot, leg2_plot, energy_bar1[0], energy_bar2[0], value_plot,

    # keep track of the visited cells for each agent
    visited_cells_agent = np.zeros((arena.sizex, arena.sizey))
    visited_cells_agent4 = np.zeros((arena.sizex, arena.sizey))

    def update(frame):
        if frame >= len(xpositions1) or frame >= len(xpositions2):
            return leg1_plot, leg2_plot, value_plot,
        # update agent positions and energies
        leg1_plot.set_data([xpositions1[frame] + 0.5], [ypositions1[frame] + 0.5])
        leg2_plot.set_data([xpositions2[frame] + 0.5], [ypositions2[frame] + 0.5])
        energy_bar1[0].set_width(u_states1[frame])
        energy_bar2[0].set_width(u_states2[frame])

        # update visited cells
        visited_cells_agent[xpositions1[frame], ypositions1[frame]] = 1
        visited_cells_agent4[xpositions2[frame], ypositions2[frame]] = 1
        for i in range(arena.sizex):
            for j in range(arena.sizey):
                if visited_cells_agent[i, j] == 1:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color=(1, 0.5, 0), alpha=0.3))
                if visited_cells_agent4[i, j] == 1:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color=(0.5, 0.5, 1), alpha=0.3))
        
        # update the value plot
        value_plot.set_data(range(frame + 1), values[:frame + 1])
        ax_value.set_xlim(0, max(1, frame))  # Adjust x-axis to current frame

        # show food sources
        for i in range(len(arena.foodsource_locs)):
            ax.scatter([arena.foodsource_locs[i][0] + 0.5], [arena.foodsource_locs[i][1] + 0.5], 
                       s=arena.foodsource_mags[i] * 3, color='green', marker='D')

        return leg1_plot, leg2_plot, value_plot,

    ani = animation.FuncAnimation(fig, update, frames=len(xpositions1), init_func=init, blit=True, interval=100)
    ani.save(filename + '.gif', writer='pillow')
    plt.tight_layout()
    plt.show()


def simulate_share(arena, xpositions1, ypositions1, xpositions2, ypositions2, u_states, values, filename):
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    ax_value = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])
    ax.set_aspect('equal')

    # initialize agents and energy plots
    leg1_plot, = ax.plot([], [], '^', color='orange', markersize=10)
    leg2_plot, = ax.plot([], [], '^', color='blue', markersize=10)
    energy_bar = ax_energy.barh([1], [0], color='purple', height=0.5)
    ax_energy.set_xlim(0, max(u_states) * 1.2)  # Added extra space on the side
    ax_energy.set_ylim(0, 2)
    ax_energy.set_yticks([1])
    ax_energy.set_yticklabels(['Shared Energy'])
    ax_energy.set_xlabel('Energy')

    # initialize a plot element for the value
    value_plot, = ax_value.plot([], [], color='blue')
    ax_value.set_xlim(0, len(values))
    ax_value.set_ylim(min(values), max(values))
    ax_value.set_xlabel('Time')
    ax_value.set_ylabel('Value')

    def init():
        ax.set_xlim(0, arena.sizex)
        ax.set_ylim(0, arena.sizey)
        ax.set_xticks(np.arange(0, arena.sizex + 1, 1))
        ax.set_yticks(np.arange(0, arena.sizey + 1, 1))
        ax.grid(True)
        return leg1_plot, leg2_plot, energy_bar[0], value_plot,

    # keep track of the visited cells for each agent
    visited_cells_agent1 = np.zeros((arena.sizex, arena.sizey))
    visited_cells_agent2 = np.zeros((arena.sizex, arena.sizey))

    def update(frame):
        if frame >= len(xpositions1) or frame >= len(xpositions2):
            return leg1_plot, leg2_plot, value_plot,
    
        # update agent positions and energies and energy
        leg1_plot.set_data([xpositions1[frame] + 0.5], [ypositions1[frame] + 0.5])
        leg2_plot.set_data([xpositions2[frame] + 0.5], [ypositions2[frame] + 0.5])
        energy_bar[0].set_width(u_states[frame])

        # update visited cells
        visited_cells_agent1[xpositions1[frame], ypositions1[frame]] = 1
        visited_cells_agent2[xpositions2[frame], ypositions2[frame]] = 1
        for i in range(arena.sizex):
            for j in range(arena.sizey):
                if visited_cells_agent1[i, j] == 1:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color=(1, 0.5, 0), alpha=0.3))
                if visited_cells_agent2[i, j] == 1:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color=(0.5, 0.5, 1), alpha=0.3))
        
        # update the value plot
        value_plot.set_data(range(frame + 1), values[:frame + 1])
        ax_value.set_xlim(0, max(1, frame))  # Adjust x-axis to current frame

         # show food sources 
        for i in range(len(arena.foodsource_locs)):
            ax.scatter([arena.foodsource_locs[i][0] + 0.5], [arena.foodsource_locs[i][1] + 0.5], 
                       s=arena.foodsource_mags[i] * 3, color='green', marker='D')

        return leg1_plot, leg2_plot, energy_bar[0], value_plot,

    ani = animation.FuncAnimation(fig, update, frames=len(xpositions1), init_func=init, blit=True, interval=100)
    ani.save(filename + '.gif', writer='pillow')
    plt.tight_layout()
    plt.show()

def heatmap_MI(xpos1, ypos1, xpos2, ypos2, mi_av, time,filename):
    # initialize the figure with 3 subplots: two for the heatmap of each agent and one for the mutual information
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    ax1.set_xlim(0, 11)
    ax1.set_ylim(0, 5)
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    heatmap1, xedges, yedges = np.histogram2d(
        [x[0] + 0.5 for x in xpos1],
        [y[0] + 0.5 for y in ypos1],
        bins=[np.arange(0, 12), np.arange(0, 6)],
        density=True
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im1 = ax1.imshow(heatmap1.T, origin='lower', cmap='bone', interpolation='nearest', extent=extent, vmin=0, vmax=0.45)

    cbar1 = fig.colorbar(im1, ax=ax1)

    heatmap2, xedges, yedges = np.histogram2d(
        [x[0] + 0.5 for x in xpos2],
        [y[0] + 0.5 for y in ypos2],
        bins=[np.arange(0, 12), np.arange(0, 6)],
        density=True
    )
    im2 = ax2.imshow(heatmap2.T, origin='lower', cmap='hot', interpolation='nearest', extent=extent, vmin=0, vmax=0.45)
    cbar2 = fig.colorbar(im2, ax=ax2)

    line, = ax3.plot([], [], color='darkblue')
    ax3.set_xlim(0, 250)
    #ax3.set_ylim(0, 0.01)
    ax3.set_ylim(0, 0.45)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mutual Information')

    # update the plots for each frame
    def update(frame):
        heatmap1, _, _ = np.histogram2d(
            [x[frame] + 0.5 for x in xpos1],
            [y[frame] + 0.5 for y in ypos1],
            bins=[np.arange(0, 12), np.arange(0, 6)],
            density=True
        )
        im1.set_data(heatmap1.T)
        im1.set_clim(0, 0.2)  

        heatmap2, _, _ = np.histogram2d(
            [x[frame] + 0.5 for x in xpos2],
            [y[frame] + 0.5 for y in ypos2],
            bins=[np.arange(0, 12), np.arange(0, 6)],
            density=True
        )
        im2.set_data(heatmap2.T)
        im2.set_clim(0, 0.2)  

        # update mutual information plot
        line.set_data(time[:frame + 1], mi_av[:frame + 1])

        return im1, im2, line

    ani = animation.FuncAnimation(fig, update, frames=len(xpos1[0]), interval=100)
    ani.save(filename+'.gif', writer='pillow')
    plt.show()