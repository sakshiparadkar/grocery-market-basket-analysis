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
