import pandas as pd
import matplotlib.pyplot as plt

# Load the data
no_bridge_df = pd.read_csv('LR_10_18_no_bridge.csv')
with_bridge_df = pd.read_csv('LR_10_18_with_bridge.csv')

# Plot of the experiment without the bridge
no_bridge_routes = no_bridge_df.groupby(['Round', 'Route']).size().unstack().fillna(0)

plt.figure(figsize=(12, 6))
plt.plot(no_bridge_routes.index, no_bridge_routes['O-L-D'], marker='o', label='OLD')
plt.plot(no_bridge_routes.index, no_bridge_routes['O-R-D'], marker='o', label='ORD')
plt.xlabel('Round Number')
plt.ylabel('Number of Subjects')
plt.title('Number of Subjects per Route (No Bridge)')
plt.legend()
plt.grid(True)
plt.show()

# Plot of the experiment with the bridge
with_bridge_routes = with_bridge_df.groupby(['Round', 'Route']).size().unstack().fillna(0)

plt.figure(figsize=(12, 6))
plt.plot(with_bridge_routes.index, with_bridge_routes['O-L-D'], marker='o', label='OLD')
plt.plot(with_bridge_routes.index, with_bridge_routes['O-R-D'], marker='o', label='ORD')
plt.plot(with_bridge_routes.index, with_bridge_routes['O-L-R-D'], marker='o', label='OLRD')
plt.xlabel('Round Number')
plt.ylabel('Number of Subjects')
plt.title('Number of Subjects per Route (With Bridge)')
plt.legend()
plt.grid(True)
plt.show()

# Payoff comparison between experiments
no_bridge_df['Payoff'] = 400 - no_bridge_df['Cost']
with_bridge_df['Payoff'] = 400 - with_bridge_df['Cost']

no_bridge_payoff = no_bridge_df.groupby('Round')['Payoff'].mean()
with_bridge_payoff = with_bridge_df.groupby('Round')['Payoff'].mean()

plt.figure(figsize=(12, 6))
plt.plot(no_bridge_payoff.index, no_bridge_payoff, marker='o', label='No Bridge')
plt.plot(with_bridge_payoff.index, with_bridge_payoff, marker='o', label='With Bridge')
plt.xlabel('Round Number')
plt.ylabel('Mean Payoff')
plt.title('Mean Payoff Comparison Between Experiments')
plt.legend()
plt.grid(True)
plt.show()

# Number of switches plot
def count_switches(df):
    switches = []
    for round_num in df['Round'].unique()[1:]:
        prev_round = df[df['Round'] == round_num - 1].set_index('Agent')['Route']
        current_round = df[df['Round'] == round_num].set_index('Agent')['Route']
        switches.append((current_round != prev_round).sum())
    return switches

no_bridge_switches = count_switches(no_bridge_df)
with_bridge_switches = count_switches(with_bridge_df)

rounds = no_bridge_df['Round'].unique()[1:]

plt.figure(figsize=(12, 6))
plt.plot(rounds, no_bridge_switches, marker='o', label='No Bridge')
plt.plot(rounds, with_bridge_switches, marker='o', label='With Bridge')
plt.xlabel('Round Number')
plt.ylabel('Number of Switches')
plt.title('Number of Switches Between Rounds')
plt.legend()
plt.grid(True)
plt.show()
