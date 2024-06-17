import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def simulate(arena, xpositions, ypositions, u_states, values, filename):
    # Create a new figure with a GridSpec layout to control the subplot positions
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # Create subplots
    ax = fig.add_subplot(gs[:, 0])
    ax_value = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])

    # Set the aspect ratio of the plot to make the cells look like squares
    ax.set_aspect('equal')

    # Initialize a plot element for the agent
    agent_plot, = ax.plot([], [], '^', color='blue', markersize=10)

    # Initialize energy bars
    energy_bar = ax_energy.barh([1], [0], color='blue', height=0.5)

    # Set the limits and labels for the energy bar plot
    ax_energy.set_xlim(0, 50)  # Limit the x-axis to 50
    ax_energy.set_ylim(0, 2)
    ax_energy.set_yticks([1])
    ax_energy.set_yticklabels(['Agent'],fontsize=20)
    ax_energy.set_xlabel('Energy', fontsize=20)

    # Initialize a plot element for the value
    value_plot, = ax_value.plot([], [], color='blue')

    # Set the limits and labels for the value plot
    ax_value.set_xlim(0, len(values))
    ax_value.set_ylim(min(values), max(values))
    ax_value.set_xlabel('Time', fontsize=20)
    ax_value.set_ylabel('Value',fontsize=20)

    def init():
        ax.set_xlim(0, arena.sizex)
        ax.set_ylim(0, arena.sizey)
        ax.set_xticks(np.arange(0, arena.sizex + 1, 1))
        ax.set_yticks(np.arange(0, arena.sizey + 1, 1))
        ax.grid(True)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # Remove ticks and labels
        value_plot.set_data(range(len(values)), values)  # Set the full range for the value plot initially
        return agent_plot, energy_bar[0], value_plot,

    # Create an array to keep track of the visited cells
    visited_cells_agent = np.zeros((arena.sizex, arena.sizey))

    def update(frame):
        # Ensure the frame index is within bounds
        if frame >= len(xpositions):
            return agent_plot, energy_bar[0], value_plot,

        # Update agent positions
        agent_plot.set_data([xpositions[frame] + 0.5], [ypositions[frame] + 0.5])

        # Update energy bar
        energy_bar[0].set_width(u_states[frame])

        # Update value plot
        value_plot.set_data(range(frame + 1), values[:frame + 1])

        # Update visited cells
        visited_cells_agent[xpositions[frame], ypositions[frame]] = 1

        # Update the color of the visited cells for the agent
        for i in range(arena.sizex):
            for j in range(arena.sizey):
                if visited_cells_agent[i, j] == 1:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color=(0.5, 0.5, 1), alpha=0.3))

        # Update reward locations
        for i in range(len(arena.foodsource_locs)):
            ax.scatter([arena.foodsource_locs[i][0] + 0.5], [arena.foodsource_locs[i][1] + 0.5], 
                    s=arena.foodsource_mags[i] * 6, color='green', marker='D')

        return agent_plot, energy_bar[0], value_plot,

    # Create the animation with a faster interval
    ani = animation.FuncAnimation(fig, update, frames=len(xpositions), init_func=init, blit=True, interval=100)

    # Save the animation
    ani.save(filename + '.gif', writer='pillow')

    # Display the animation
    plt.tight_layout()
    plt.show()
    

def simulate_greedy(arena, xpositions1, ypositions1, filename):
    fig = plt.figure(figsize=(6, 6))    
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    leg_plot, = ax.plot([], [], '^', color='purple', markersize=10)

    def init():
        ax.set_xlim(0, arena.sizex)
        ax.set_ylim(0, arena.sizey)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.vlines(np.arange(0, arena.sizex + 1), 0, arena.sizey, colors='black', linestyles='solid', linewidth=0.2)
        ax.hlines(np.arange(0, arena.sizey + 1), 0, arena.sizex, colors='black', linestyles='solid', linewidth=0.2)
        return leg_plot,

    # Create separate arrays to keep track of the visited cells for each agent
    visited_cells_agent = np.zeros((arena.sizex, arena.sizey))

    # draw visited cells
    def update(frame):
        if frame >= len(xpositions1):
            return leg_plot,
        leg_plot.set_data([xpositions1[frame] + 0.5], [ypositions1[frame] + 0.5])
        visited_cells_agent[xpositions1[frame], ypositions1[frame]] = 1
        for i in range(arena.sizex):
            for j in range(arena.sizey):
                if visited_cells_agent[i, j] == 1:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color=(0.7, 0.5, 1), alpha=0.3))

        for i in range(len(arena.foodsource_locs)):
            ax.scatter([arena.foodsource_locs[i][0] + 0.5], [arena.foodsource_locs[i][1] + 0.5], 
                       s=arena.foodsource_mags[i] * 3, color='green', marker='D')

        return leg_plot,
    
    ani = animation.FuncAnimation(fig, update, frames=len(xpositions1), init_func=init, blit=True, interval=100)
    ani.save(filename + '.gif', writer='pillow')
    #plt.tight_layout()
    #plt.show()

    