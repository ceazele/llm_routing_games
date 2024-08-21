import pandas as pd
import matplotlib.pyplot as plt

# Load the data
no_bridge_df = pd.read_csv('COT_10_18/routes_10_18/routes_no_bridge_10_18.csv')
with_bridge_df = pd.read_csv('COT_10_18/routes_10_18/routes_with_bridge_10_18.csv')

# Define a function to map routes to numbers for plotting
route_mapping = {'O-A-D': 1, 'O-B-D': 2, 'O-A-B-D': 3}

# Apply the mapping to the data
no_bridge_df['Route_Num'] = no_bridge_df['Route'].map(route_mapping)
with_bridge_df['Route_Num'] = with_bridge_df['Route'].map(route_mapping)

# Function to plot agent trajectories
def plot_agent_trajectories(df, title):
    plt.figure(figsize=(12, 8))
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        plt.plot(agent_data['Round'], agent_data['Route_Num'], marker='o', label=f'Agent {agent}')
    plt.xlabel('Round Number')
    plt.ylabel('Route')
    plt.yticks(list(route_mapping.values()), list(route_mapping.keys()))
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()

# Plot trajectories for no bridge
plot_agent_trajectories(no_bridge_df, 'Agent Trajectories (No Bridge)')

# Plot trajectories for with bridge
plot_agent_trajectories(with_bridge_df, 'Agent Trajectories (With Bridge)')
