import textwrap
import pandas as pd
import matplotlib.pyplot as plt

year = 2025
#Read gestion_coordinaciones.xlsx
df = pd.read_excel(f'gestion_coordinaciones_{year}.xlsx', sheet_name='consolidado')

#Print the first 5 rows of the dataframe
print(df.head())

# Pivot the data for the stacked bar chart
pivot_df = df.pivot(index="Funciones y Atribuciones", columns="Sede", values="Avance")

# Set the font to DejaVu Sans (Ubuntu alternative)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman No9 L', 'Times New Roman', 'DejaVu Serif']

# Wrap y-axis labels if necessary
pivot_df.index = [ '\n'.join(textwrap.wrap(label, width=80)) for label in pivot_df.index ]

# Increase figure width to accommodate text
fig, ax = plt.subplots(figsize=(15, 12))  # Increase width

# Remove upper, right, and bottom spines (borders)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Plot stacked bar chart
colors = ['#F2C180', '#F6F190', '#C5EBA0']
pivot_df.plot(kind="barh", stacked=True, ax=ax, color=colors)

# Add labels for each avance for each Sede
for i, (index, row) in enumerate(pivot_df.iterrows()):
   cumulative = 0
   for sede in pivot_df.columns:
      avance = row[sede]
      if avance > 0:  # Only label if avance is greater than 0
         ax.text(cumulative + avance / 2, i, f"{avance*100:.0f}%", ha="center", va="center", fontsize=13, color="black")
      cumulative += avance

# Customize the plot
plt.title("Avance de Funciones y Atribuciones por Sede", fontsize=16, weight='bold')
plt.xlabel("Avance", fontsize=16, weight='bold')
plt.ylabel("Funciones y Atribuciones", fontsize=16, weight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Ensure Y-axis labels are fully visible on one line
ax.set_yticklabels(pivot_df.index, fontsize=14, ha="right")  

# Increase left margin to fit long labels
plt.subplots_adjust(left=0.4)

# Bold legend title
legend = plt.legend(title="Sede", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=14)
plt.setp(legend.get_title(), weight='bold')  # Set legend title to bold

# Show the plot
plt.tight_layout()

# Hide x-axis ticks
plt.xticks([])

plt.savefig(f'coordinaciones_{year}.svg', format='svg')
plt.savefig(f'coordinaciones_{year}.png', format='png')
