
# =====================================================================================================
# Business Problem
# =====================================================================================================

# Armut, which is Turkey's largest online service platform, brings together service providers and those who want to
# receive services. It enables easy access to services such as cleaning, renovation, and transportation with just a few
# touches on a computer or smartphone. An association rule learning-based product recommendation system is desired
# to be created using the dataset containing users who received services and the categories of services they received.

# =====================================================================================================
# Story of the Dataset
# =====================================================================================================
# The dataset consists of the services customers receive and the categories of these services. Date and time of each
# service received contains information.

# UserId: Customer ID
# ServiceId: They are anonymized services belonging to each category. (Example: Upholstery washing service under
# Cleaning category) A ServiceId can be found under different categories and represents different services under
# different categories. (Example: The service with CategoryId 7 and ServiceId 4 is honeycomb cleaning, while the
# service with CategoryId 2 and ServiceId 4 is furniture assembly)
# CategoryId: They are anonymized categories. (Example: Cleaning, transportation, renovation category)
# CreateDate: The date the service was purchased


# =====================================================================================================
# TASK 1: Preparing the Data
# =====================================================================================================
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import warnings


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)

# Step 1: Read your armut_data.csv file.

main_df = pd.read_csv('Recommendation_Systems/armut_data.csv')
df = main_df.copy()


def check_dataframe(df, row_num=5):
    print("*************** Dataset Shape ***************")
    print("No. of Rows:", df.shape[0], "\nNo. of Columns:", df.shape[1])
    print("*************** Dataset Information ***************")
    print(df.info())
    print("*************** Types of Columns ***************")
    print(df.dtypes)
    print(f"*************** First {row_num} Rows ***************")
    print(df.head(row_num))
    print(f"*************** Last {row_num} Rows ***************")
    print(df.tail(row_num))
    print("*************** Summary Statistics of The Dataset ***************")
    print(df.describe().T)
    print("*************** Dataset Missing Values Analysis ***************")
    print(df.isnull().sum())
    print("*************** Dataset Duplicates ***************")
    print(df[df.duplicated() == True])


check_dataframe(df)

# Step 2: ServiceId represents a different service for each CategoryId. Combine ServiceId and CategoryId
# with "_" to create a new variable to represent the services.

# Step 3: The dataset consists of services purchased by customers with the date and time of purchase, but there is
# no basket definition (invoice, etc.). In order to apply Association Rule Learning, a basket definition needs to be
# created, which represents the services purchased by each customer on a monthly basis. For example, customer with
# ID 7256 has a basket consisting of services 9_4 and 46_4 purchased in August 2017, and a different basket consisting
# of services 9_4 and 38_4 purchased in October 2017. The baskets should be identified with a unique ID. To do this,
# first create a new date variable that only includes the year and month. Then combine UserId and the new date variable
# using "_" and assign it to a new variable named BasketId.


def data_prep(df):
    df['Service'] = df['ServiceId'].astype(str) + '_' + df['CategoryId'].astype(str)
    df['CreateDate'] = pd.to_datetime(df['CreateDate'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%d-%m-%Y %H:%M:%S')
    df['New_Date'] = pd.to_datetime(df['CreateDate']).dt.strftime('%Y-%m')
    df['BasketId'] = df['UserId'].astype(str) + '_' + df['New_Date']
    return df


data_prep(df)

# =====================================================================================================
# TASK 2: Create Association Rules
# =====================================================================================================

# Step 1: Create a pivot table with BasketId values in rows and Service values in columns.
# Step 2: Create association rules.
# Step 3: Step 6: Using the arl_recommender function, recommend a service to a user who has received the last
# 2_0 service.

# Service         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# BasketId
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..


def prep_for_arl_recommender(df):
    # # Create a new dataframe where each row represents a basket and each column represents a service
    # Fill missing values with 0
    # Convert the count of each category in a basket to binary values (1 if the count is > 0, else 0)
    apriori_df = df.groupby(['BasketId', 'Service'])['CategoryId'].count().unstack().fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)

    # Find frequent itemsets that occur in at least min_support (0.01) of the baskets
    frequent_itemsets = apriori(apriori_df, min_support=0.01, use_colnames=True, low_memory=True)

    # Generate association rules from the frequent itemsets, using a minimum support threshold of 0.01
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


rules = prep_for_arl_recommender(df)
rules


def arl_recommender(rules_df, service, rec_count=1):
    # Sort the rules based on lift in descending order
    sorted_rules = rules_df.sort_values("lift", ascending=False)

    # Filter the rules where the antecedent contains the given service
    filtered_rules = sorted_rules[sorted_rules["antecedents"].apply(lambda x: service in x)]

    # Extract the recommended services from the filtered rules
    recommendation_list = [', '.join(rule) for rule in filtered_rules["consequents"].apply(list).tolist()[:rec_count]]

    print("Other recommended services for the service you entered: ", recommendation_list)


arl_recommender(rules, '2_0')

