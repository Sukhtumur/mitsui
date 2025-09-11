import polars as pl
import pandas as pd
import sys

# Test private phase simulation
print("Testing private phase...")

# Create simulated future test data
test_df = pd.read_csv('data/test.csv')
test_df['date_id'] = test_df['date_id'] + 200  # Shift to future dates
test_pl = pl.from_pandas(test_df)

print(f'Simulated private test data: {test_pl.shape}')
print(f'Future date range: {test_df["date_id"].min()} to {test_df["date_id"].max()}')

# Test prediction
from mitsui_final_submission import predict

result = predict(test_pl)
result_pd = result.to_pandas()

# Validate
required_cols = ['date_id'] + [f'target_{i}' for i in range(424)]
missing = [c for c in required_cols if c not in result_pd.columns]
print(f'Result shape: {result.shape}')
print(f'Missing columns: {len(missing)}')

# Check ranges
target_cols = [f'target_{i}' for i in range(424)]
stats = result_pd[target_cols].describe()
min_val = stats.loc['min'].min()
max_val = stats.loc['max'].max()
print(f'Prediction range: [{min_val:.4f}, {max_val:.4f}]')

if len(missing) == 0 and abs(max_val) <= 0.15:
    print('✅ PRIVATE PHASE VALIDATION PASSED!')
else:
    print('❌ Issue detected')
