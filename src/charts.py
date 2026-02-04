from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# Global style
plt.rcParams["font.family"] = "Nimbus Roman"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# Alternative version that accepts custom colors
def get_gradient_cmap(start_hex="#003772", end_hex="#FCC000", num_colors=256):
   """
   Generate a list of colors forming a gradient between start and end colors.
   
   Args:
      start_hex (str): Start color in hex format (e.g., "#003772")
      end_hex (str): End color in hex format (e.g., "#FCC000")
      num_colors (int): Total number of colors to generate (includes start and end)
   
   Returns:
      list: List of hex color codes
   """
   # Remove the '#' if present
   start_hex = start_hex.lstrip('#')
   end_hex = end_hex.lstrip('#')
   
   # Convert hex to RGB
   start_rgb = [int(start_hex[i:i+2], 16) for i in (0, 2, 4)]
   end_rgb = [int(end_hex[i:i+2], 16) for i in (0, 2, 4)]
   
   # Generate intermediate colors
   colors = []
   for i in range(num_colors):
      if num_colors == 1:
            ratio = 0
      else:
            ratio = i / (num_colors - 1)
      
      # Interpolate each RGB component
      r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
      g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
      b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
      
      # Convert back to hex
      hex_color = f"#{r:02x}{g:02x}{b:02x}"
      colors.append(hex_color.upper())
   
   return colors

def __save_figure(fig: plt.Figure, filename_base: str):
   """Save figure in both SVG and PNG formats."""
   filename_base = filename_base.lower().replace(" ", "_").replace(".", "_")
   fig.savefig(f"{filename_base}.svg", format="svg", bbox_inches="tight")
   fig.savefig(f"{filename_base}.png", format="png", dpi=300, bbox_inches="tight")
   fig.savefig(f"{filename_base}.pdf", format="pdf", bbox_inches="tight")

def plot_distribution_by_group(df: pd.DataFrame, 
                              group_col: str, 
                              category_col: str, 
                              value_col: str, 
                              kind="bar",
                              threshold_pct=10,
                              fig_size_x=6,
                              fig_size_y_per_group=4,
                              grouped_others_min_count=3):
   """
   For each group (e.g., Campus), plot distribution of categories as subplot.
   Categories with percentage <= threshold are grouped as 'Otros'.
   """
   groups = df[group_col].unique()
   n = len(groups)
   fig, axes = plt.subplots(n, 1, figsize=(fig_size_x, fig_size_y_per_group * n))

   if n == 1:
      axes = [axes]

   for ax, group in zip(axes, groups):
      sub = df[df[group_col] == group].copy()
      
      # Calculate percentages
      total = sub[value_col].sum()
      sub['pct'] = sub[value_col] / total * 100
      
      if len(sub) > grouped_others_min_count:
         # Group small categories into 'Otros'
         mask = sub['pct'] <= threshold_pct
         if mask.any():
            others_sum = sub.loc[mask, value_col].sum()
            sub = sub.loc[~mask]
            # Use pd.concat instead of append
            otros_row = pd.DataFrame([{category_col: "Otros", value_col: others_sum, 'pct': others_sum/total*100}])
            sub = pd.concat([sub, otros_row], ignore_index=True)

      sub = sub.set_index(category_col)
      
      colors = get_gradient_cmap(num_colors=len(sub))
      
      if kind == "bar":
         bars = ax.bar(sub.index, sub[value_col], color=colors)
         #Remove top and right borders
         ax.spines['top'].set_visible(False)
         ax.spines['right'].set_visible(False)
         ax.text(-0.2, 0.5, f"{group}", transform=ax.transAxes,
               rotation=90, va='center', ha='right',
               fontsize=12, fontweight="bold", clip_on=False)
         ax.set_ylabel(f"{value_col}", fontsize=10, fontweight="regular", labelpad=10)
         ax.grid(axis="y", linestyle=":", linewidth=1, color="gray", alpha=0.4)

         # Label on top of bars
         for bar, val, pct in zip(bars, sub[value_col], sub['pct']):
               ax.text(bar.get_x() + bar.get_width()/2, val + total*0.01,
                     f"{val} ({pct:.1f}%)",
                     ha='center', va='bottom', fontsize=10, rotation=0)

      elif kind == "pie":
         # Labels with value and percentage on a new line
         labels = [f"{cat}\n{int(val)} ({pct:.1f}%)"
               for cat, val, pct in zip(sub.index, sub[value_col], sub['pct'])]

         ax.pie(sub[value_col], labels=labels, colors=colors,
            textprops={'fontsize': 14})
         ax.text(-0.2, 0.5, f"{group}", transform=ax.transAxes,
               rotation=90, va='center', ha='right',
               fontsize=16, fontweight="bold", clip_on=False)

   plt.tight_layout(h_pad=0.5)
   #Save figure into svg and png
   fig_name = f'output/d_chart_{group_col}_{category_col}_{value_col}'
   __save_figure(fig, fig_name)
   plt.close()


def plot_distribution_by_two_groups(df: pd.DataFrame, 
                                    group_col: str, 
                                    subgroup_col: str, 
                                    category_col: str, 
                                    value_col: str, 
                                    kind: str = "bar",
                                    threshold_pct: float = 10,
                                    grouped_others_min_count=3):
   """
   Generic Case 2: For each group (e.g., Campus) and subgroup (e.g., Género),
   create subplot of category distribution.
   """
   groups = df[group_col].unique()
   subgroups = df[subgroup_col].unique()

   fig, axes = plt.subplots(len(groups), len(subgroups), figsize=(6 * len(subgroups), 4 * len(groups)))

   if len(groups) == 1:
      axes = [axes]
   if len(subgroups) == 1:
      axes = [[ax] for ax in axes]

   y_group_text = -0.25 if kind == "pie" else -0.15
   for i, group in enumerate(groups):
      #Add title to axes[i]
      axes[i][0].text(y_group_text, 0.5, f"{group}", transform=axes[i][0].transAxes,
               rotation=90, va='center', ha='right',
               fontsize=16, fontweight="bold", clip_on=False)
      for j, subgroup in enumerate(subgroups):
         ax = axes[i][j]
         sub = df[(df[group_col] == group) & (df[subgroup_col] == subgroup)]
         if sub.empty:
               ax.axis("off")
               continue
         
         # Calculate percentages
         total = sub[value_col].sum()
         sub = sub.copy()
         sub['pct'] = sub[value_col] / total * 100
         
         if len(sub) > grouped_others_min_count:
            # Group small categories into 'Otros'
            mask = sub['pct'] <= threshold_pct
            if mask.any():
               others_sum = sub.loc[mask, value_col].sum()
               sub = sub.loc[~mask]
               # Use pd.concat instead of append
               otros_row = pd.DataFrame([{category_col: "Otros", value_col: others_sum, 'pct': others_sum/total*100}])
               sub = pd.concat([sub, otros_row], ignore_index=True)

         sub = sub.set_index(category_col)[value_col]
         colors = get_gradient_cmap(num_colors=len(sub))

         if kind == "bar":
            bars = ax.bar(sub.index, sub.values, color=colors)
            #Set fontsize of x ticks
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontweight="regular")
            #Set fontsize of y ticks
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, fontweight="regular")
            #Remove top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis="y", linestyle=":", linewidth=1, color="gray", alpha=0.4)

            total = sub.sum()
            for bar, val in zip(bars, sub.values):
               ax.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + total*0.01,
                     f"{int(val)} ({val/total*100:.1f}%)",
                     ha="center", va="bottom", fontsize=12)
            if j == 0:
               ax.set_ylabel(f"{value_col}", fontsize=12, fontweight="regular", labelpad=10)
            if i == len(groups) - 1:
               ax.set_xlabel(f"{subgroup}", fontsize=14, fontweight="bold", labelpad=20)

         elif kind == "pie":
            total = sub.sum()
            labels = [f"{cat}\n{int(val)} ({val/total*100:.1f}%)" for cat, val in zip(sub.index, sub.values)]
            ax.pie(sub.values, labels=labels, colors=colors, textprops={'fontsize': 15})
            
            if i == len(groups) - 1:
               ax.set_xlabel(f"{subgroup}", fontsize=16, fontweight="bold", labelpad=20)

   plt.tight_layout(pad=1.0, h_pad=3.0)

   #Save into png and svg
   fig_name = f'output/dist_chart_{group_col}_{subgroup_col}_{category_col}'
   __save_figure(fig, fig_name)
   plt.close()

def plot_grouped_barchart(df: pd.DataFrame, 
                        group_col: str, 
                        category_col: str, 
                        value_col: str, 
                        xlabel: str = None, 
                        ylabel: str = None):
   """
   Plots a grouped bar chart with value and percentage annotations.
   
   Parameters:
   - df: pandas DataFrame
   - group_col: column name for grouping (e.g., 'Sede')
   - category_col: column name for categories (e.g., 'IRI 2025')
   - value_col: column name for values (e.g., 'Docentes')
   - title, xlabel, ylabel: optional labels
   """
   
   # Set font to Nimbus Roman (or fallback)
   plt.rcParams['font.family'] = ['Nimbus Roman', 'serif']
   
   # Calculate percentages within each group
   df = df.copy()
   df['Total_by_Group'] = df.groupby(group_col)[value_col].transform('sum')
   df['Percentage'] = (df[value_col] / df['Total_by_Group']) * 100
   
   # Pivot for plotting
   pivot = df.pivot(index=group_col, columns=category_col, values=value_col)
   pivot_percent = df.pivot(index=group_col, columns=category_col, values='Percentage')
   
   # Plot setup
   fig, ax = plt.subplots(figsize=(10, 6))

   #Remove border top and right
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   
   # Bar positions
   n_groups = len(pivot.index)
   n_categories = len(pivot.columns)
   bar_width = 0.8 / n_categories
   index = range(n_groups)
   
   # Colors (optional: use default or specify)
   colors = get_gradient_cmap(num_colors=n_categories)
   
   # Plot each category
   for i, (cat, color) in enumerate(zip(pivot.columns, colors)):
      bars = ax.bar(
         [x + i * bar_width for x in index],
         pivot[cat],
         bar_width,
         label=cat,
         color=color
      )
      
      # Annotate each bar with value and percentage
      for j, (bar, pct) in enumerate(zip(bars, pivot_percent[cat])):
         height = bar.get_height()
         if height > 0:
               ax.text(
                  bar.get_x() + bar.get_width() / 2,
                  height + max(pivot.max()) * 0.01,  # slight offset
                  f'{int(height)}\n({pct:.1f}%)',
                  ha='center', va='bottom',
                  fontsize=9
               )
   
   # Labels and ticks
   if xlabel:
      ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
   ax.set_ylabel(ylabel or value_col, fontsize=11, fontweight='regular')
   ax.set_xticks([x + bar_width * (n_categories - 1) / 2 for x in index])
   ax.set_xticklabels(pivot.index, fontsize=10, fontweight='bold')
   
   # Legend
   ax.legend(title=category_col)
   
   # Adjust layout
   plt.tight_layout()
   
   #Save into png and svg
   fig_name = f'output/g_bar_chart_{group_col}_{category_col}_{value_col}'
   __save_figure(fig, fig_name)
   plt.close()

def plot_bar_chart(df: pd.DataFrame,
                  group_col: str,
                  category_col: str, 
                  value_col: str, 
                  percentage_col: str,
                  percentage_as_value: bool = False,
                  fig_size_x=6,
                  fig_size_y_per_group=4):
   # Set font to Nimbus Roman (or fallback)
   plt.rcParams['font.family'] = ['Nimbus Roman', 'serif']

   groups = df[group_col].unique()
   n = len(groups)
   fig, axes = plt.subplots(n, 1, figsize=(fig_size_x, fig_size_y_per_group * n))

   if n == 1:
      axes = [axes]

   for ax, group in zip(axes, groups):
      sub = df[df[group_col] == group].copy()
      sub = sub.set_index(category_col)
      colors = get_gradient_cmap(num_colors=len(sub))
      if percentage_as_value:
         bars = ax.bar(sub.index, sub[percentage_col], color=colors)
      else:
         bars = ax.bar(sub.index, sub[value_col], color=colors)
      #Remove top and right borders
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.text(-0.2, 0.5, f"{group}", transform=ax.transAxes,
            rotation=90, va='center', ha='right',
            fontsize=12, fontweight="bold", clip_on=False)
      #ax.set_ylabel(f"{value_col}", fontsize=10, fontweight="regular", labelpad=10)
      ax.grid(axis="y", linestyle=":", linewidth=1, color="gray", alpha=0.4)
      # Label on top of bars
      for bar, val, pct in zip(bars, sub[value_col], sub[percentage_col]):
         if percentage_as_value:
            ax.text(bar.get_x() + bar.get_width()/2, pct + sub[percentage_col].sum()*0.01,
               f"{pct:.1f}% ({val})",
               ha='center', va='bottom', fontsize=10, rotation=0)
         else:
            ax.text(bar.get_x() + bar.get_width()/2, val + sub[value_col].sum()*0.01,
                  f"{val} ({pct:.1f}%)",
                  ha='center', va='bottom', fontsize=10, rotation=0)
   
   plt.tight_layout()
   #Save into png and svg
   fig_name = f'output/bar_chart_{category_col}_{value_col}'
   __save_figure(fig, fig_name)
   plt.close()

def wrap_labels(labels, max_width):
   wrapped_labels = []
   for label in labels:
      wrapped_label = '\n'.join([label[i:i+max_width] for i in range(0, len(label), max_width)])
      wrapped_labels.append(wrapped_label)
   return wrapped_labels

def plot_scatter_chart(
   df: pd.DataFrame,
   x_col: str,
   y_col: str,
   size_col: str,
   label_col: str,
   bubble_factor: float = 35,
   jitter: float = 0.05,
   label_y_threshold: int = 5,
   label_x_threshold: int = 4
):
   """
   Improved scatter/bubble plot with:
   - Outlier-aware scaling
   - Intelligent label selection
   - Non-linear bubble sizing
   - adjustText label collision avoidance
   """

   # ---------- Setup ----------
   fig, ax = plt.subplots(figsize=(13, 8))

   # Sort so large bubbles go behind
   df = df.sort_values(size_col, ascending=False)

   # Color map
   # cmap = LinearSegmentedColormap.from_list(
   #    "custom_gradient", ["#16467B", "#FDCD5B"]
   # )
   # colors = cmap(np.linspace(0, 1, len(df)))

   #Colors tab20 palette
   colors = plt.get_cmap("tab20").colors
   colors = [colors[i % len(colors)] for i in range(len(df))]

   texts = []

   # ---------- Plot points ----------
   for i, (_, row) in enumerate(df.iterrows()):
      x = row[x_col]
      y = row[y_col]

      # Conditional jitter (dense area only)
      if x <= 3 and y <= 10:
         x += np.random.normal(0, jitter)
         y += np.random.normal(0, jitter)

      # Intelligent labeling rule
      label_this = (
         (row[y_col] >= label_y_threshold) or
         (row[x_col] >= label_x_threshold)
      )

      ax.scatter(
         x,
         y,
         s=np.sqrt(row[size_col]) * bubble_factor,
         color=colors[i],
         alpha=0.85 if label_this else 0.45,
         edgecolors="none",
         zorder=3
      )

      if label_this:
         texts.append(
               ax.text(
                  x,
                  y,
                  row[label_col],
                  fontsize=6,
                  alpha=0.9
               )
         )

   # ---------- Label adjustment ----------
   adjust_text(
      texts,
      ax=ax,
      only_move={'points': 'y', 'texts': 'y'},
      expand_points=(1.2, 1.4),
      expand_text=(1.2, 1.4),
      arrowprops=dict(
         arrowstyle='-',
         color='gray',
         lw=0.5,
         alpha=0.5
      )
   )

   # ---------- Styling ----------
   ax.spines["top"].set_visible(False)
   ax.spines["right"].set_visible(False)

   ax.set_xlabel(x_col, fontsize=11)
   ax.set_ylabel(y_col, fontsize=11)

   ax.grid(True, linestyle="--", alpha=0.15)

   # Visual guide zones (optional but helpful)
   ax.axvspan(0, 3, alpha=0.03, color="gray")
   ax.axhspan(0, 10, alpha=0.03, color="gray")

   plt.tight_layout()

   # ---------- Save ----------
   #Save into png and svg
   fig_name = f'output/scatter_chart_{x_col}_{y_col}'
   __save_figure(fig, fig_name)
   plt.close()

def plot_scatter_chart_broken_axis(
   df: pd.DataFrame,
   x_col: str,
   y_col: str,
   size_col: str,
   label_col: str,
   bubble_factor: float = 35,
   jitter: float = 0.05,
   label_y_threshold: int = 4,
   label_x_threshold: int = 2,
   y_break_low: int = 22,
   y_break_high: int = 45
):
   """
   Scatter/Bubble plot with:
   - Broken Y-axis for extreme outliers
   - Intelligent label selection
   - Collision-free labels
   - Static-report quality styling
   """

   # ---------- Figure & Axes ----------
   fig, (ax_top, ax_bottom) = plt.subplots(
      2, 1,
      sharex=True,
      figsize=(13, 9),
      gridspec_kw={"height_ratios": [0.7, 4], "hspace": 0.05}
   )

   # Sort so large bubbles go behind
   df = df.sort_values(size_col, ascending=False)

   # Colormap
   # cmap = LinearSegmentedColormap.from_list(
   #    "custom_gradient", ["#16467B", "#FDCD5B"]
   # )
   # colors = cmap(np.linspace(0, 1, len(df)))
   #Colors tab20 palette
   colors = plt.get_cmap("tab20").colors
   colors = [colors[i % len(colors)] for i in range(len(df))]

   texts_top = []
   texts_bottom = []

   # ---------- Plot ----------
   for i, (_, row) in enumerate(df.iterrows()):
      x = row[x_col]
      y = row[y_col]

      # Conditional jitter (dense area only)
      if x <= 3 and y <= 10:
         x += np.random.normal(0, jitter)
         y += np.random.normal(0, jitter)

      label_this = (
         (row[y_col] >= label_y_threshold) or
         (row[x_col] >= label_x_threshold)
      )

      target_ax = ax_top if y >= y_break_high else ax_bottom

      target_ax.scatter(
         x,
         y,
         s=np.sqrt(row[size_col]) * bubble_factor,
         color=colors[i],
         alpha=0.85 if label_this else 0.45,
         edgecolors="none",
         zorder=3
      )

      if label_this:
         if target_ax is ax_top:
            texts_top.append(ax_top.text(x, y, row[label_col], fontsize=8))
         else:
            texts_bottom.append(ax_bottom.text(x, y, row[label_col], fontsize=8))

   # ---------- Y-axis limits ----------
   ax_bottom.set_ylim(0, y_break_low)
   ax_top.set_ylim(y_break_high, df[y_col].max() * 1.05)

   # ---------- Broken axis markers ----------
   ax_top.spines["bottom"].set_visible(False)
   ax_bottom.spines["top"].set_visible(False)
   ax_top.tick_params(labeltop=False)
   ax_bottom.xaxis.tick_bottom()

   d = 0.008
   kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
   ax_top.plot((-d, +d), (-d, +d), **kwargs)
   ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

   kwargs.update(transform=ax_bottom.transAxes)
   ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
   ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

   # ---------- Adjust labels ----------
   for txt in texts_bottom:
      txt.set_clip_on(True)

   for txt in texts_top:
      txt.set_clip_on(True)
   adjust_text(
      texts_bottom,
      ax=ax_bottom,
      only_move={"points": "y", "texts": "y"},
      expand_points=(1.2, 1.4),
      expand_text=(1.2, 1.4),
      arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5)
   )
   adjust_text(
      texts_top,
      ax=ax_top,
      only_move={"points": "y", "texts": "y"},
      expand_points=(1.2, 1.4),
      expand_text=(1.2, 1.4),
      arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5)
   )

   # ---------- Styling ----------
   for ax in (ax_top, ax_bottom):
      ax.grid(True, linestyle="--", alpha=0.15)
      ax.spines["right"].set_visible(False)
      ax.spines["top"].set_visible(False)

   ax_bottom.set_xlabel(x_col, fontsize=11)
   ax_bottom.set_ylabel(y_col, fontsize=11)

   fig.subplots_adjust(top=0.88)

   # ---------- Save ----------
   #Save into png and svg
   fig_name = f'output/scatter_chart_{x_col}_{y_col}'
   __save_figure(fig, fig_name)
   plt.close()

def plot_pie_chart(df: pd.DataFrame,
                  category_col: str,
                  value_col: str,
                  figsize_x: int = 6,
                  figsize_y: int = 6,
                  fig_name: str = None):
   plt.figure(figsize=(figsize_x, figsize_y))
   # Prepare data
   labels = df[category_col]
   sizes = df[value_col]
   
   # Colors
   colors = get_gradient_cmap(num_colors=len(df))
   
   # Plot pie chart
   wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors,
                                    autopct='%1.1f%%', startangle=140,
                                    textprops={'fontsize': 16})
   
   # Styling
   for text in texts:
      text.set_fontsize(14)
   for autotext in autotexts:
      autotext.set_fontsize(16)
      autotext.set_color('white')
      autotext.set_fontweight('bold')
   
   plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
   plt.tight_layout()
   #Save into png, svg and eps
   fig_name = f'output/{fig_name}' if fig_name else f'output/pie_chart_{category_col}_{value_col}'
   fig = plt.gcf()
   __save_figure(fig, fig_name)
   plt.close()

def plot_vertical_bar_chart(df: pd.DataFrame,
                        category_col: str,
                        value_col: str,
                        figsize_x: int = 8,
                        figsize_y: int = 6):
   plt.figure(figsize=(figsize_x, figsize_y))
   
   # Sort dataframe by value_col descending
   df_sorted = df.sort_values(by=value_col, ascending=False)
   
   # Colors
   colors = get_gradient_cmap(num_colors=len(df_sorted))
   
   # Plot bar chart
   bars = plt.bar(df_sorted[category_col], df_sorted[value_col], color=colors)
   
   # Add value labels on top of bars
   for bar in bars:
      height = bar.get_height()
      plt.text(bar.get_x() + bar.get_width() / 2, height + max(df_sorted[value_col]) * 0.01,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=10)
   
   # Styling
   plt.xlabel(category_col, fontsize=12, fontweight='bold')
   plt.ylabel(value_col, fontsize=12, fontweight='bold')
   plt.xticks(rotation=45, ha='right', fontsize=10)
   plt.grid(axis='y', linestyle=':', alpha=0.7)
   plt.gca().spines['top'].set_visible(False)
   plt.gca().spines['right'].set_visible(False)
   
   plt.tight_layout()
   #Save into png, svg and eps
   fig_name = f'output/{fig_name}' if fig_name else f'output/v_bar_chart_{category_col}_{value_col}'
   fig = plt.gcf()
   __save_figure(fig, fig_name)
   plt.close()
   
def plot_horizontal_bar_chart(df: pd.DataFrame,
                        category_col: str,
                        value_col: str,
                        figsize_x: int = 8,
                        figsize_y: int = 6,
                        fig_name: str = None):
   plt.figure(figsize=(figsize_x, figsize_y))
   
   # Sort dataframe by value_col descending
   df_sorted = df.sort_values(by=value_col, ascending=True)
   
   # Colors
   colors = get_gradient_cmap(num_colors=len(df_sorted))
   
   # Plot horizontal bar chart
   bars = plt.barh(df_sorted[category_col], df_sorted[value_col], color=colors)
   
   # Add value labels at the end of bars
   for bar in bars:
      width = bar.get_width()
      plt.text(width + max(df_sorted[value_col]) * 0.01, bar.get_y() + bar.get_height() / 2,
               f'{int(width)}',
               ha='left', va='center', fontsize=10)
   
   # Styling
   plt.ylabel(category_col, fontsize=12, fontweight='bold')
   plt.xlabel(value_col, fontsize=12, fontweight='bold')
   plt.yticks(fontsize=10)
   plt.grid(axis='x', linestyle=':', alpha=0.7)
   plt.gca().spines['top'].set_visible(False)
   plt.gca().spines['right'].set_visible(False)
   
   plt.tight_layout()
   #Save into png, svg and eps
   fig_name = f'output/{fig_name}' if fig_name else f'output/h_bar_chart_{category_col}_{value_col}'
   fig = plt.gcf()
   __save_figure(fig, fig_name)
   plt.close()

def plot_lines_chart(df: pd.DataFrame,
                     x_col: str,
                     values_col: str,
                     x_label: str = None,
                     y_label: str = None,
                     linewidth: int = 2,
                     markersize: int = 6,
                     color: str = '#003772',
                     figsize_x: int = 8,
                     figsize_y: int = 5,
                     y_steps: int = None,
                     dynamic_ylim: bool = False,
                     partial_data: bool = True,
                     fig_name: str = None):
   plt.figure(figsize=(figsize_x, figsize_y))
   if partial_data:
      # 1. Separar los datos: todos menos el último, y los dos últimos
      complete_df = df.iloc[:-1]
      partial_df = df.iloc[-2:]
      # 2. Dibujar la línea sólida (Hasta 2024)
      plt.plot(complete_df[x_col], complete_df[values_col], 
               marker='o', linestyle='-', linewidth=linewidth, 
               color=color, markersize=markersize, clip_on=False)
      # 3. Dibujar la línea punteada para el segmento final (2024 a 2025)
      plt.plot(partial_df[x_col], partial_df[values_col], 
               marker='o', linestyle='--', linewidth=linewidth, 
               color=color, alpha=0.8, markersize=markersize, clip_on=False)
   else:
      plt.plot(df[x_col], df[values_col], marker='o', linestyle='-', 
            linewidth=linewidth, markersize=markersize, color=color, clip_on=False)
   
   # Set font of yticks and xticks
   plt.xticks(fontsize=14, fontweight='regular')
   plt.yticks(fontsize=14, fontweight='regular')
   
   #Configurate xlim
   plt.xlim(df[x_col].min(), df[x_col].max())
   x_col_values = df[x_col].astype(int).tolist()
   labels = [str(a) for a in x_col_values]
   # Add asterisk to last label if partial data
   if partial_data:
      labels[-1] = f"{labels[-1]}*"

   plt.xticks(x_col_values, labels)
   plt.xlim(df[x_col].min() - 0.2, df[x_col].max() + 0.2)
   
   # Set labels
   plt.xlabel(x_col if x_label is None else x_label, fontsize=14, fontweight='bold')
   plt.ylabel(values_col if y_label is None else y_label, fontsize=14, fontweight='bold')
   # Add padding into x_label and y_label
   plt.xlabel(plt.gca().get_xlabel(), labelpad=15)
   plt.ylabel(plt.gca().get_ylabel(), labelpad=15)
   # Set dynamic ylim
   if dynamic_ylim:
      max_val = df[values_col].max()
      plt.ylim(0, max_val * 1.1)
      #Set steps of y axis
      if y_steps:
         plt.yticks(np.arange(0, max_val * 1.1, y_steps))
   

   plt.grid(axis='y', linestyle=':', alpha=0.7)
   plt.gca().spines['top'].set_visible(False)
   plt.gca().spines['right'].set_visible(False)
   plt.tight_layout()
   #Save into png, svg and eps
   fig_name = f'output/{fig_name}' if fig_name else f'output/line_chart_{x_col}_{values_col}'
   fig = plt.gcf()
   __save_figure(fig, fig_name)
   plt.close()

def plot_lines_chart_pivot(df: pd.DataFrame,
                     category_col: str,
                     x_label: str = None,
                     y_label: str = None,
                     figsize_x: int = 6,
                     figsize_y: int = 4,
                     partial_data: bool = True,
                     fig_name: str = None):
   #1. Set the category column as the index so .loc[category] works
   df_plot = df.set_index(category_col)
   
   # Identify year columns only (exclude the index/category name)
   x_values = df_plot.columns
   x_indices  = range(len(x_values))
   
   plt.figure(figsize=(figsize_x, figsize_y))
   markers = 'o'

   categories = df_plot.index.unique()
   n_categories = len(categories)
   
   #2. Assuming get_gradient_cmap is defined elsewhere in your helper utils
   colors_list = get_gradient_cmap(num_colors=n_categories)
   colors_dict = {cat: colors_list[i] for i, cat in enumerate(categories)}

   # 3. Plot each row
   for category in categories:
      # Use df_plot.loc to get the row values for the years
      if partial_data:
         # Plot solid line for all but last point
         plt.plot(x_indices[:-1], df_plot.loc[category].iloc[:-1], 
                  marker=markers, 
                  label=category, 
                  color=colors_dict[category], 
                  linewidth=2.5, 
                  markersize=10,
                  clip_on=False)
         # Plot dashed line for last segment
         plt.plot(x_indices[-2:], df_plot.loc[category].iloc[-2:], 
                  marker=markers, 
                  label=None,  # No label for dashed segment
                  color=colors_dict[category], 
                  linewidth=2.5, 
                  linestyle='--', 
                  alpha=0.8,
                  markersize=10,
                  clip_on=False)
      else:
         plt.plot(x_indices, df_plot.loc[category], 
               marker=markers, 
               label=category, 
               color=colors_dict[category], 
               linewidth=2.5, 
               markersize=10,
               clip_on=False)

   # 4. Formatting
   if y_label is not None:
      plt.ylabel(y_label, fontsize=12, fontweight='bold')
   if x_label is not None:
      plt.xlabel(x_label, fontsize=12, fontweight='bold')
   
   # Dynamic ylim: Adjust based on data max to avoid cutting off lines
   max_val = df_plot.max().max()
   plt.ylim(0, max_val * 1.1)
   
   # X-axis ticks
   x_values = [str(x) for x in x_values]
   if partial_data:
      x_values[-1] = f"{x_values[-1]}*"
   plt.xticks(x_indices, x_values, fontsize=10)

   plt.legend(loc='center left', bbox_to_anchor=(1, 0.7), frameon=False, fontsize=12)
   plt.grid(axis='y', linestyle=':', alpha=0.7)
   plt.gca().spines['top'].set_visible(False)
   plt.gca().spines['right'].set_visible(False)

   plt.tight_layout()
   #Save into png, svg and eps
   fig_name = f'output/{fig_name}' if fig_name else f'output/line_chart_pivot_{category_col}_{x_indices}'
   fig = plt.gcf()
   __save_figure(fig, fig_name)
   plt.close()

def plot_stacked_bar_chart(df: pd.DataFrame,
                        index_col: str,
                        x_label: str,
                        y_label: str,
                        figsize_x: int = 8,
                        figsize_y: int = 6,
                        rotation_x_ticks: int = 0,
                        custom_colors: bool = False,
                        fig_name: str = None):
   plt.figure(figsize=(figsize_x, figsize_y))
   
   df.set_index(index_col, inplace=True)

   # Colors
   cols = df.columns.tolist()
   colors = get_gradient_cmap(num_colors=len(df.columns)) if custom_colors else None
   
   # 2. Create the plot
   # Using a colormap like 'tab20' helps distinguish between 11 different categories
   ax = df.plot(kind='bar', stacked=True, figsize=(10, 7), 
               color=colors, 
               colormap='tab20' if not custom_colors else None,
               edgecolor='white')

   # 3. Customization
   plt.xlabel(x_label, fontsize=15, fontweight='bold', labelpad=15)
   plt.ylabel(y_label, fontsize=15, fontweight='bold', labelpad=15)
   #Set fontsize of x ticks and y ticks
   plt.xticks(fontsize=14, fontweight='regular', rotation=rotation_x_ticks)
   plt.yticks(fontsize=14, fontweight='regular')

   plt.grid(axis='y', linestyle=':', alpha=0.7)
   plt.gca().spines['top'].set_visible(False)
   plt.gca().spines['right'].set_visible(False)

   #Legend
   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

   plt.tight_layout()
   #Save into png, svg and eps
   fig_name = f'output/{fig_name}' if fig_name else f'output/stack_bar_chart_{index_col}_{cols}'
   fig = plt.gcf()
   __save_figure(fig, fig_name)
   plt.close()

def plot_dynamic_bars(df: pd.DataFrame, 
                     index_col: str,
                     y_label: str,
                     x_label: str = None,
                     figsize_x: int = 8,
                     figsize_y: int = 6,
                     rotation_x_ticks: int = 0,
                     fig_name: str = None):
   # 1. Prepare Data: Set the chosen column as index
   # We create a copy so we don't modify the original dataframe outside the function
   plot_df = df.set_index(index_col).copy()
   
   # 2. Setup Dimensions
   categories = plot_df.index
   cols = plot_df.columns
   n_cols = len(cols)
   
   x = np.arange(len(categories)) 
   # Total width for all bars in a group is 0.8; divide by number of bars
   width = 0.8 / n_cols  
   
   # Custom Palette
   colors = get_gradient_cmap(num_colors=n_cols)
   
   # 3. Create Plot
   fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

   # 4. Generate Bars
   for i, col in enumerate(cols):
      # Math to center the group of bars
      offset = (i - (n_cols - 1) / 2) * width
      
      rects = ax.bar(x + offset, plot_df[col], width, 
                     label=str(col), 
                     color=colors[i % len(colors)])
      
      # Add the numeric labels on top
      #ax.bar_label(rects, padding=3, color='white', fontweight='bold', fontsize=10)

   # 3. Customization
   if x_label:
      plt.xlabel(x_label, fontsize=15, fontweight='bold', labelpad=15)
   plt.ylabel(y_label, fontsize=15, fontweight='bold', labelpad=15)

   plt.xticks(rotation=rotation_x_ticks)
   ax.set_xticks(x)
   ax.set_xticklabels(categories, fontsize=12)
   
   # Clean up axes
   ax.tick_params(axis='y')
   ax.yaxis.grid(True, linestyle='--', alpha=0.2, color='gray')
   ax.set_axisbelow(True)
   
   for spine in ['top', 'right']:
      ax.spines[spine].set_visible(False)

   # Legend
   legend = ax.legend(
      frameon=False, 
      loc='upper left', 
      bbox_to_anchor=(1, 1),
      fontsize=10  # Sets font size for all legend text
   )

   plt.tight_layout()

   #Save into png, svg and eps
   fig_name = f'output/{fig_name}' if fig_name else f'output/dynamic_bar_chart_{index_col}_{cols}'
   __save_figure(fig, fig_name)
   plt.close()
