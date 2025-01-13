import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df=pd.read_csv('feature.csv',sep=',',index_col='UNIQUEID')
print(df.info())

category_cols = ['MANUFACTURER_ID', 'STATE_ID', 'DISBURSAL_MONTH', 'DISBURSED_CAT', 'PERFORM_CNS_SCORE_DESCRIPTION', 'EMPLOYMENT_TYPE']
print(df[category_cols].dtypes)

df[category_cols] = df[category_cols].astype('category')
print(df[category_cols].dtypes)

small_cols = ['STATE_ID', 'LTV', 'DISBURSED_CAT', 'PERFORM_CNS_SCORE', 'DISBURSAL_MONTH', 'LOAN_DEFAULT']

loan_df_sml = df[small_cols]

small_cols = ['STATE_ID', 'LTV', 'DISBURSED_CAT', 'PERFORM_CNS_SCORE', 'DISBURSAL_MONTH', 'LOAN_DEFAULT']

df_sml = df[small_cols]

print(df_sml.shape)

print(df_sml.info())

x = df_sml.drop(['LOAN_DEFAULT'], axis=1)
y = df_sml['LOAN_DEFAULT']

print("x has {0} rows and {1} columns".format(x.shape[0], x.shape[1]))
print("y has {0} rows".format(y.count()))

print(x.info())
print(y.dtype)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("x_train has {0} rows and {1} columns".format(x_train.shape[0], x_train.shape[1]))
print("x_test has {0} rows and {1} columns".format(x_test.shape[0], x_test.shape[1]))
print("y_train has {0} rows".format(y_train.count()))
print("y_test has {0} rows".format(y_test.count()))

print(x_train.info())

print(y_train.head())

print(x_test.info())

print(y_test.head())


print(y_train.value_counts(normalize=True))

print(y_test.value_counts(normalize=True))



data_dumm = pd.get_dummies(loan_df_sml, prefix_sep='_', drop_first=True)

print(data_dumm.info())


print(data_dumm['STATE_ID_13'].value_counts(normalize=True))


print(data_dumm['DISBURSAL_MONTH_10'].value_counts(normalize=True))


x = data_dumm.drop(['LOAN_DEFAULT'], axis=1)
y = data_dumm['LOAN_DEFAULT']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))



logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(x_train, y_train)
print(logistic_model)

preds = logistic_model.predict(x_test)
print(preds)

print(logistic_model.score(x_test, y_test))