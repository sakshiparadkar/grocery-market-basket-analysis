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
