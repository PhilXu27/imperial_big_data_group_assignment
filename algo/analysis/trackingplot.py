import pandas as pd
import matplotlib.pyplot as plt

df_new = pd.read_csv("passive_portfolio_performance.csv", header=None)
custom_labels = ['25', '50', '100', '250', '500', '2.5k', '5k', '10k']
tracking_error = pd.to_numeric(df_new.iloc[1, 2:len(custom_labels) + 2])
baseline_tracking_error = pd.to_numeric(df_new.iloc[1, 1])
plt.figure(figsize=(8, 6))
plt.scatter(custom_labels, tracking_error, color='blue', alpha=0.7, label="Tracking Error")
plt.axhline(y=baseline_tracking_error, color='red', linestyle='dashed', label="Baseline Tracking Error")
plt.xlabel('Portfolio Size')
plt.ylabel('Tracking Error')
plt.title('Tracking Error vs Portfolio Size')
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
