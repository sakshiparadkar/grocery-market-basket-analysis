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
