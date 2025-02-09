import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")  # or "white", or any style you prefer

fig, ax = plt.subplots(figsize=(10, 6))

# Hide the actual x and y axes (we'll place text manually)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# -- Title at the top
ax.text(0.5, 9.5,
        "Mean regret of LLM decisions by representation method\n"
        "in selfish routing game, within framework for\n"
        "informativeness of history representations",
        ha='center', va='top', fontsize=12, fontweight='bold')

# -- Horizontal arrow for "representation of chat history"
ax.annotate(
    "Representation of chat history",
    xy=(1.0, 8.5), xytext=(9.0, 8.5),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    va='center', ha='left',
    fontsize=10, fontweight='regular'
)

# -- Label above the arrow showing "Compression of natural language..."
ax.text(5, 8.8,
        "Compression of natural language\n(distilling relevant history information)",
        ha='center', va='bottom', fontsize=9, style='italic')

# -- Vertical arrow for "action informativity"
ax.annotate(
    "Action\ninformativity",
    xy=(0.5, 1.0), xytext=(0.5, 7.5),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    va='bottom', ha='center',
    fontsize=10
)

# -- Horizontal axis labels for "Payoff" (left) → "Regret" (right)
ax.text(3.0, 2.0, "Payoff", ha='center', va='center', fontsize=10)
ax.text(7.0, 2.0, "Regret", ha='center', va='center', fontsize=10)
ax.text(5.0, 1.5, "Reward informativity", ha='center', va='center', fontsize=9)

# -- Left region: "Full chat"
ax.text(2.0, 8.2, "Full chat\nrepresentation of chat history",
        ha='center', va='top', color='#D55E00', fontsize=10, fontweight='bold')

# Place the three data points in the left region
# (x ~ 2, adjust y’s to stack them nicely)
ax.text(2, 6.3, "APO\n113.76", ha='center', va='center', fontsize=10)
ax.text(2, 4.8, "AP\n56.87", ha='center', va='center', fontsize=10)
ax.text(2, 3.3, "AR\n0.86", ha='center', va='center', fontsize=10)

# -- Right region: "Summary"
ax.text(8.0, 8.2, "Summary\nrepresentation of chat history",
        ha='center', va='top', color='#D55E00', fontsize=10, fontweight='bold')

# Place the three data points in the right region
# (x ~ 8, adjust y’s to stack them similarly)
ax.text(8, 6.3, "APO\n26.22", ha='center', va='center', fontsize=10)
ax.text(8, 4.8, "AP\n6.54", ha='center', va='center', fontsize=10)
ax.text(8, 3.3, "AR\n0.98", ha='center', va='center', fontsize=10)

# -- Extra annotations on y-axis: "Own action" (bottom) → "Others' actions + own action" (top)
ax.text(0.8, 2.5, "Own action", ha='right', va='center', fontsize=9)
ax.text(0.8, 6.5, "Others' actions + own action", ha='right', va='center', fontsize=9)

plt.tight_layout()
plt.show()
