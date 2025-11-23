#!/usr/bin/env python3
"""
Generate blocksize speedup plot from blocksize_speedup_data.txt
Replaces the gnuplot script with a Python/matplotlib implementation
"""

import matplotlib.pyplot as plt
from collections import defaultdict

# Read data from file
data = []
with open('blocksize_speedup_data.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                i_max = float(parts[0])
                blocksize = float(parts[1])
                speedup = float(parts[2])
                t_max = float(parts[3])
                data.append((i_max, blocksize, speedup, t_max))

# Organize data by (i_max, t_max) combinations
grouped_data = defaultdict(list)
for i_max, blocksize, speedup, t_max in data:
    key = (i_max, t_max)
    grouped_data[key].append((blocksize, speedup))

# Sort keys for consistent ordering
sorted_keys = sorted(grouped_data.keys())

# Color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot data for each (i_max, t_max) combination
for idx, (i_max, t_max) in enumerate(sorted_keys):
    # Get data for this combination and sort by blocksize
    plot_data = grouped_data[(i_max, t_max)]
    plot_data.sort(key=lambda x: x[0])
    blocksize_vals = [x[0] for x in plot_data]
    speedup_vals = [x[1] for x in plot_data]

    # Plot with lines and points
    color = colors[idx % len(colors)]
    ax.plot(blocksize_vals, speedup_vals,
            marker='o',
            markersize=6,
            linewidth=1.5,
            color=color,
            label=f'i_max = {int(i_max):,}, t_max = {int(t_max):,}')

# Set x-axis to log scale base 2
ax.set_xscale('log', base=2)

# Set custom ticks for powers of 2 from 2^5 to 2^10
tick_values = [2**i for i in range(5, 11)]  # 32, 64, 128, 256, 512, 1024
tick_labels = [str(2**i) for i in range(5, 11)]  # Show actual values
ax.set_xticks(tick_values)
ax.set_xticklabels(tick_labels)

# Labels and title
ax.set_xlabel('Blocksize', fontsize=12, fontname='Arial')
ax.set_ylabel('Speedup', fontsize=12, fontname='Arial')
ax.set_title('Blocksize Speedup (Unoptimized): Time_{32} / Time_{blocksize}',
             fontsize=14, fontname='Arial', pad=10)

# Grid
ax.grid(True, alpha=0.3)

# Legend
ax.legend(loc='best', fontsize=10)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('blocksize_speedup_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved as blocksize_speedup_plot.png")

# Optionally display
# plt.show()

