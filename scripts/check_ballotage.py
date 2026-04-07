import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/circuitos_merged.parquet')
DEST = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']
Y_sum = df[DEST].sum(axis=1)
total = df['total_ballotage']
diff = total - Y_sum
print("total_ballotage vs sum(dest_cols):")
print(f"  max diff: {diff.max():.0f}")
print(f"  min diff: {diff.min():.0f}")
print(f"  mean diff: {diff.mean():.1f}")
print(f"  circuits where total < Y_sum: {(diff < 0).sum()}")
print(f"  circuits where diff > 0: {(diff > 0).sum()}")
print(f"  circuits where diff == 0: {(diff == 0).sum()}")
print(f"\nSample Y_sum vs total:")
print(df[['total_ballotage'] + DEST].head(5).assign(Y_sum=Y_sum))
