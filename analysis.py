# ── CELL 1 : Install Libraries ──────────────────────────────
# !pip install mlxtend wordcloud networkx


# ── CELL 2 : Imports & Theme Setup ──────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from wordcloud import WordCloud
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Dark purple theme
plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#444',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#aaa',
    'ytick.color':      '#aaa',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2a3e',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
})

PURPLE = '#7b2ff7'
PINK   = '#f72585'
CYAN   = '#4cc9f0'
ORANGE = '#f8961e'
ACCENT = ['#7b2ff7', '#f72585', '#4cc9f0', '#f8961e', '#43aa8b', '#90e0ef']


# ── CELL 3 : Load & Explore Data ────────────────────────────
df = pd.read_csv('Groceries_dataset.csv')

df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['itemDescription'] = df['itemDescription'].str.strip().str.lower()

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values :", df.isnull().sum().sum())
print("Unique items   :", df['itemDescription'].nunique())
print("Unique customers:", df['Member_number'].nunique())

# Build transactions
transactions = df.groupby('Member_number')['itemDescription'].apply(list).tolist()
print(f"\n✅ Transactions ready: {len(transactions)}")



# ── CELL 4 : Plot 1 — Top 20 Most Purchased Items ───────────
item_counts = df['itemDescription'].value_counts().head(20)

fig, ax = plt.subplots(figsize=(12, 7))
colors = [ACCENT[i % len(ACCENT)] for i in range(20)]
ax.barh(item_counts.index[::-1], item_counts.values[::-1],
        color=colors, edgecolor='none', height=0.7)

for i, val in enumerate(item_counts.values[::-1]):
    ax.text(val + 30, i, f'{val:,}', va='center', fontsize=8.5, color='#ccc')

ax.set_xlabel('Number of Purchases')
ax.set_title('Top 20 Most Purchased Grocery Items', fontsize=15, fontweight='bold', pad=15)
ax.grid(axis='x')
plt.tight_layout()
plt.savefig('plot1_top20_items.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()


# ── CELL 5 : Plot 2 — Monthly Transactions Trend ────────────
monthly = df.groupby('Month').size().reset_index(name='count')
monthly['Month_str'] = monthly['Month'].astype(str)
x = range(len(monthly))

fig, ax = plt.subplots(figsize=(15, 5))
ax.fill_between(x, monthly['count'], alpha=0.2, color=PURPLE)
ax.plot(x, monthly['count'], color=PURPLE, linewidth=2.5, marker='o', markersize=5)

# Annotate peak
peak_idx = monthly['count'].idxmax()
ax.annotate(f"📈 Peak\n{monthly.loc[peak_idx, 'count']} transactions",
            xy=(peak_idx, monthly.loc[peak_idx, 'count']),
            xytext=(peak_idx + 1, monthly.loc[peak_idx, 'count'] + 50),
            arrowprops=dict(arrowstyle='->', color=PINK, lw=1.5),
            fontsize=9, color=PINK)

# Annotate lowest
low_idx = monthly['count'].idxmin()
ax.annotate(f"📉 Low\n{monthly.loc[low_idx, 'count']}",
            xy=(low_idx, monthly.loc[low_idx, 'count']),
            xytext=(low_idx + 1, monthly.loc[low_idx, 'count'] - 80),
            arrowprops=dict(arrowstyle='->', color=CYAN, lw=1.5),
            fontsize=9, color=CYAN)

ax.set_xticks(list(x))
ax.set_xticklabels(monthly['Month_str'], rotation=45, ha='right', fontsize=8)
ax.set_title('Monthly Transaction Volume — Trend Analysis',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Month')
ax.set_ylabel('Number of Transactions')
ax.grid(axis='y')
plt.tight_layout()
plt.savefig('monthly_trend.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()


# ── CELL 6 : Plot 3 — Basket Size Histogram + Box Plot ──────
basket_sizes = [len(t) for t in transactions]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(basket_sizes, bins=30, color=CYAN, edgecolor='#0f0f1a', alpha=0.85)
axes[0].axvline(np.mean(basket_sizes), color=PINK, linewidth=2,
                label=f'Mean = {np.mean(basket_sizes):.1f}')
axes[0].axvline(np.median(basket_sizes), color=ORANGE, linewidth=2,
                linestyle='--', label=f'Median = {int(np.median(basket_sizes))}')
axes[0].set_title('Basket Size — Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Items per Transaction')
axes[0].set_ylabel('Frequency')
axes[0].legend(fontsize=9)
axes[0].grid(axis='y')

# Box plot
axes[1].boxplot(basket_sizes, patch_artist=True,
                boxprops=dict(facecolor=PURPLE, alpha=0.6, color=PURPLE),
                medianprops=dict(color=PINK, linewidth=2.5),
                whiskerprops=dict(color='#aaa'),
                capprops=dict(color='#aaa'),
                flierprops=dict(marker='o', color=CYAN, alpha=0.4, markersize=3))
axes[1].set_title('Basket Size — Box Plot', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Items per Transaction')
axes[1].grid(axis='y')

fig.suptitle('How Many Items Do Customers Buy Per Visit?',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('basket_size.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

# ── CELL 7 : Plot 8 — Top 20 Association Rules Bar Chart ───
top20 = strong_rules.head(20).copy()
top20['rule'] = top20['antecedents_str'] + '  →  ' + top20['consequents_str']

colors = plt.cm.plasma(np.linspace(0.15, 0.92, 20))
fig, ax = plt.subplots(figsize=(13, 9))

bars = ax.barh(top20['rule'][::-1], top20['lift'][::-1],
               color=colors[::-1], edgecolor='none', height=0.65)

for bar, val in zip(bars, top20['lift'][::-1]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', fontsize=8.5, color='white')

ax.axvline(1, color='#aaa', linestyle='--', linewidth=1.2, label='Baseline (Lift = 1)')
ax.set_title('Top 20 Association Rules — Ranked by Lift',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Lift Value', fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis='x')
plt.tight_layout()
plt.savefig('plot8_top20_rules.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

# ── CELL 7 : Plot 8 — Top 20 Association Rules Bar Chart ───
top20 = strong_rules.head(20).copy()
top20['rule'] = top20['antecedents_str'] + '  →  ' + top20['consequents_str']

colors = plt.cm.plasma(np.linspace(0.15, 0.92, 20))
fig, ax = plt.subplots(figsize=(13, 9))

bars = ax.barh(top20['rule'][::-1], top20['lift'][::-1],
               color=colors[::-1], edgecolor='none', height=0.65)

for bar, val in zip(bars, top20['lift'][::-1]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', fontsize=8.5, color='white')

ax.axvline(1, color='#aaa', linestyle='--', linewidth=1.2, label='Baseline (Lift = 1)')
ax.set_title('Top 20 Association Rules — Ranked by Lift',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Lift Value', fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis='x')
plt.tight_layout()
plt.savefig('plot8_top20_rules.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

# ── CELL 8 : Plot 9 — Co-occurrence Heatmap ────────────────
top_items = df['itemDescription'].value_counts().head(12).index.tolist()
df_top = df[df['itemDescription'].isin(top_items)]
basket_top = df_top.groupby('Member_number')['itemDescription'].apply(list)

cooc = pd.DataFrame(0, index=top_items, columns=top_items)
for items in basket_top:
    unique_items = list(set(items))
    for i in unique_items:
        for j in unique_items:
            if i != j:
                cooc.loc[i, j] += 1

mask = np.eye(len(cooc), dtype=bool)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cooc, mask=mask, annot=True, fmt='d', cmap='magma',
            linewidths=0.5, linecolor='#0f0f1a',
            cbar_kws={'label': 'Co-occurrence Count'}, ax=ax)
ax.set_title('Top 12 Items — Co-occurrence Heatmap',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('plot9_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()


# ── CELL 9 : Plot 10 — Network Graph ───────────────────────
top_net = strong_rules.head(35)

G = nx.DiGraph()
for _, row in top_net.iterrows():
    G.add_edge(row['antecedents_str'], row['consequents_str'],
               weight=row['lift'], conf=row['confidence'])

fig, ax = plt.subplots(figsize=(18, 13))
fig.patch.set_facecolor('#0a0a14')
ax.set_facecolor('#0a0a14')

pos = nx.spring_layout(G, k=3.0, seed=42, iterations=80)

degrees     = dict(G.degree())
node_sizes  = [500 + degrees[n] * 400 for n in G.nodes()]
node_colors = [degrees[n] for n in G.nodes()]

edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
w_min, w_max = min(edge_weights), max(edge_weights)
edge_norm    = [(w - w_min) / (w_max - w_min + 1e-9) for w in edge_weights]

nx.draw_networkx_edges(G, pos, ax=ax,
    edge_color=edge_norm, edge_cmap=plt.cm.plasma,
    width=[1.5 + 3 * n for n in edge_norm],
    arrows=True, arrowsize=18,
    connectionstyle='arc3,rad=0.12', alpha=0.85)

nx.draw_networkx_nodes(G, pos, ax=ax,
    node_size=node_sizes,
    node_color=node_colors, cmap=plt.cm.cool,
    alpha=0.95)

nx.draw_networkx_labels(G, pos, ax=ax,
    font_size=8, font_color='white', font_weight='bold')

sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
     norm=plt.Normalize(w_min, w_max))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
cbar.set_label('Lift', color='white', fontsize=12)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

ax.set_title('Association Rules — Network Graph\n'
             'Node size = connections  |  Edge color = Lift strength',
             fontsize=16, fontweight='bold', color='white', pad=20)
ax.axis('off')
plt.tight_layout()
plt.savefig('plot10_network_graph.png', dpi=150, bbox_inches='tight', facecolor='#0a0a14')
plt.show()

# ── CELL 11 : Plot 7 — Support vs Lift Scatter ──────────────
fig, ax = plt.subplots(figsize=(12, 6))

sc = ax.scatter(
    rules['support'],
    rules['lift'],
    c=rules['confidence'],
    cmap='cool',
    s=50,
    alpha=0.7,
    edgecolors='none'
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Confidence', color='white', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

ax.axhline(y=1, color='#aaa', linestyle='--', linewidth=1.2, label='Lift = 1 (baseline)')
ax.set_title('Support vs Lift  (Color = Confidence)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Support', fontsize=12)
ax.set_ylabel('Lift', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True)
plt.tight_layout()
plt.savefig('plot7_support_vs_lift.png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.show()

