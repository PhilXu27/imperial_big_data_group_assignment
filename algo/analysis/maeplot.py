import pandas as pd
import matplotlib.pyplot as plt

df_new = pd.read_csv("passive_portfolio_performance.csv", header=None)
custom_labels = ['25', '50', '100', '250', '500', '2.5k', '5k', '10k']
mean_absolute_error = pd.to_numeric(df_new.iloc[3, 2:len(custom_labels) + 2])
baseline_mae = pd.to_numeric(df_new.iloc[3, 1])
plt.figure(figsize=(8, 6))
plt.scatter(custom_labels, mean_absolute_error, color='green', alpha=0.7, label="Mean Absolute Error")
plt.axhline(y=baseline_mae, color='red', linestyle='dashed', label="Baseline MAE")
plt.xlabel('Portfolio Size')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error vs Portfolio Size')
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
