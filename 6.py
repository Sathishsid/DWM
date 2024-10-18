import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the dataset
data = pd.read_csv('D:\Desktop\DWM\store_data.csv')

# Display top 5 rows of data and check columns
print("Top 5 rows of the dataset:")
print(data.head())
print("\nColumn names:", data.columns)

# Ensure the column names are correct
if 'almonds' not in data.columns or 'honey' not in data.columns:
    print("Error: 'almonds' or 'honey' column not found in the dataset.")
else:
    # Data preprocessing: converting transaction data into a binary format
    basket = (data.groupby(['almonds', 'honey'])['honey'].count().unstack().reset_index().fillna(0).set_index('almonds'))

    # Convert count values to 1 for items that were bought and 0 otherwise
    basket_sets = basket.applymap(lambda x: 1 if x >= 1 else 0)

    # Apriori algorithm to find frequent item sets
    frequent_item_sets = apriori(basket_sets, min_support=0.0045, use_colnames=True)

    # Association rules
    rules = association_rules(frequent_item_sets, metric="lift", min_threshold=3)

    # Filter rules based on confidence
    filtered_rules = rules[rules['confidence'] >= 0.2]

    # Display the association rules
    print("\nAssociation Rules with min_confidence >= 0.2, min_support=0.0045, min_lift=3, min_length=2:")
    print(filtered_rules)
