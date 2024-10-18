import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# Load the dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('D:\DWM/your_dataset.csv')
# Display top 5 rows of data
print("Top 5 rows of the dataset:")
print(data.head())
# Data preprocessing: converting transaction data into a binary format
basket = (data.groupby(['Transaction', 'Item'])['Item']
          .count().unstack().reset_index().fillna(0)
          .set_index('Transaction'))
# Convert count values to 1 for items that were bought and 0 otherwise
basket_sets = basket.applymap(lambda x: 1 if x >= 1 else 0)
# Apriori algorithm to find frequent item sets
frequent_item_sets = apriori(basket_sets, min_support=0.0045, use_colnames=True)
# Association rules
association_rules = association_rules(frequent_item_sets, metric="lift", min_threshold=3)
filtered_rules = association_rules[association_rules['confidence'] >= 0.2]
# Display the association rules
print("\nAssociation Rules with min_confidence >= 0.2, min_support=0.0045, min_lift=3, min_length=2:")
print(filtered_rules)
