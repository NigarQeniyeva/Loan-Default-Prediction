import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('analysis.csv', sep=',', index_col='UNIQUEID')

print(df.head())

def explore_categorical(df, col_name):   
    print("{0} Summary".format(col_name))
    print("\n")

    print("{0} Counts".format(col_name))
    print(df[col_name].value_counts())
    print("\n")

    print("{0} Ratio".format(col_name))
    print(df[col_name].value_counts(normalize=True))
    print("\n")

    print("{0} Default Counts".format(col_name))
    print(df.groupby(col_name)['LOAN_DEFAULT'].value_counts().unstack(level=-1))
    print("\n")

    print("{0} Default Ratio".format(col_name))
    print(df.groupby(col_name)['LOAN_DEFAULT'].value_counts(normalize=True).unstack(level=-1))
    print("\n")

    sns.catplot(data=df, kind='count', x=col_name, hue='LOAN_DEFAULT')
    plt.show()

def explore_continuous(df, col_name):
    print("{0} Summary".format(col_name))
    print("\n")
    print(df[col_name].describe())
    print("\n")

    sns.boxplot(x=col_name, data=df)
    plt.show()

    sns.displot(df[col_name], kind='hist', kde=False)
    plt.show()

    print("{0} Grouped Summary".format(col_name))
    print("\n")
    print(df.groupby('LOAN_DEFAULT')[col_name].describe())

    sns.boxplot(x=col_name, y='LOAN_DEFAULT', data=df, orient="h")
    plt.show()

explore_continuous(df, 'DISBURSED_AMOUNT')
print([df['DISBURSED_AMOUNT'].idxmax()])
print(df.loc[df['DISBURSED_AMOUNT'].idxmax()])

disbursed_buckets = [13000, 30000, 45000, 60000, 75000, 150000, 1000000]
disbursed_labels = ['13k - 30k', '30k - 45k', '45k - 60k', '60k - 75k', '75k - 150k', '150k - 1m']

df['DISBURSED_CAT'] = pd.cut(df['DISBURSED_AMOUNT'], disbursed_buckets, labels=disbursed_labels)

explore_categorical(df, 'DISBURSED_CAT')

df['DISBURSAL_DIFFERENCE'] = df['ASSET_COST'] - df['DISBURSED_AMOUNT']
print(df[['DISBURSAL_DIFFERENCE', 'ASSET_COST', 'DISBURSED_AMOUNT']].sample(5))

df['TOTAL_ACCTS'] = df['PRI_NO_OF_ACCTS'] + df['SEC_NO_OF_ACCTS']
print(df[['TOTAL_ACCTS', 'PRI_NO_OF_ACCTS', 'SEC_NO_OF_ACCTS']].sample(10))

df['TOTAL_ACTIVE_ACCTS'] = df['PRI_ACTIVE_ACCTS'] + df['SEC_ACTIVE_ACCTS']
df['TOTAL_OVERDUE_ACCTS'] = df['PRI_OVERDUE_ACCTS'] + df['SEC_OVERDUE_ACCTS']
df['TOTAL_CURRENT_BALANCE'] = df['PRI_CURRENT_BALANCE'] + df['SEC_CURRENT_BALANCE']
df['TOTAL_SANCTIONED_AMOUNT'] = df['PRI_SANCTIONED_AMOUNT'] + df['SEC_SANCTIONED_AMOUNT'] 
df['TOTAL_DISBURSED_AMOUNT'] = df['PRI_DISBURSED_AMOUNT'] + df['SEC_DISBURSED_AMOUNT']
df['TOTAL_INSTAL_AMT'] = df['PRIMARY_INSTAL_AMT'] + df['SEC_INSTAL_AMT']

drop_cols = ['PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS', 'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT', 'PRI_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT', 'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS', 'SEC_OVERDUE_ACCTS', 'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT', 'SEC_DISBURSED_AMOUNT', 'SEC_INSTAL_AMT']

df = df.drop(drop_cols, axis=1)

df['OVERDUE_PCT'] = df['TOTAL_OVERDUE_ACCTS'] / df['TOTAL_ACCTS']

print(df['OVERDUE_PCT'].isnull().sum())

df['OVERDUE_PCT'] = df['OVERDUE_PCT'].fillna(0)

print(df['OVERDUE_PCT'].isnull().sum())

print(df.info())

numeric_cols = ['DISBURSED_AMOUNT', 
                'ASSET_COST', 
                'LTV', 
                'NEW_ACCTS_IN_LAST_SIX_MONTHS', 
                'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 
                'NO_OF_INQUIRIES', 
                'AGE', 
                'AVERAGE_ACCT_AGE_MONTHS', 
                'CREDIT_HISTORY_LENGTH_MONTHS',
                'PERFORM_CNS_SCORE',
                'TOTAL_ACCTS',
                'TOTAL_ACTIVE_ACCTS',
                'TOTAL_OVERDUE_ACCTS',
                'TOTAL_CURRENT_BALANCE', 
                'TOTAL_SANCTIONED_AMOUNT', 
                'TOTAL_DISBURSED_AMOUNT', 
                'TOTAL_INSTAL_AMT', 
                'OVERDUE_PCT', 
                'DISBURSAL_DIFFERENCE']

df[['LTV', 'ASSET_COST', 'AGE']].boxplot()
plt.title('Prior to Scaling')
plt.show()

mm_scaler = MinMaxScaler()
df[numeric_cols] = mm_scaler.fit_transform(df[numeric_cols])

df[['LTV', 'ASSET_COST', 'AGE']].boxplot()
plt.title('After Scaling')
plt.show()

df.to_csv('feature.csv')