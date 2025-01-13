import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_csv("data.csv", sep=';', index_col='UNIQUEID')


print(df.head())
print(df.shape)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


print(df.info()) 

df['DISBURSAL_DATE'] = pd.to_datetime(df['DISBURSAL_DATE'], format='%d.%m.%Y')
df["DATE_OF_BIRTH"]=pd.to_datetime(df['DATE_OF_BIRTH'], format='%d.%m.%Y')


print("Max Disbursal Date:", df['DISBURSAL_DATE'].max())
print("Min Disbursal Date:", df['DISBURSAL_DATE'].min())

print('Timespan of Data Set:',df['DISBURSAL_DATE'].max()-df['DISBURSAL_DATE'].min())

print(df['LOAN_DEFAULT'].value_counts())
print(df['LOAN_DEFAULT'].value_counts(normalize=True))

sns.countplot(x="LOAN_DEFAULT",data=df)
plt.show()

print(df.isnull())
print(df.isnull().any())
print(df.isnull().sum())


df=df.fillna(value={'EMPLOYMENT_TYPE':'Missing'})

print(df['EMPLOYMENT_TYPE'].value_counts())

sns.countplot(x="EMPLOYMENT_TYPE",data=df)
plt.show()

print(df[['DISBURSAL_DATE','DATE_OF_BIRTH']].sample(10))

df['AGE']=df['DISBURSAL_DATE']-df['DATE_OF_BIRTH']

df['AGE']=df['AGE'].dt.days // 365


print(df[['DATE_OF_BIRTH','AGE','DISBURSAL_DATE']].sample(n=10))

df['DISBURSAL_MONTH']=df['DISBURSAL_DATE'].dt.month
print(df['DISBURSAL_MONTH'].value_counts())

df=df.drop(['DISBURSAL_DATE','DATE_OF_BIRTH'],axis=1)
print(df[['CREDIT_HISTORY_LENGTH','AVERAGE_ACCT_AGE']].sample(n=10))


def calc_months(str_list):
    years=int(str_list[0])
    months=int(str_list[1])

    num_months=(years * 12)+ months
    return num_months

df['AVERAGE_ACCT_AGE_MONTHS']=df['AVERAGE_ACCT_AGE'].str.findall('\d+')
print(df['AVERAGE_ACCT_AGE_MONTHS'].sample(10))

df['AVERAGE_ACCT_AGE_MONTHS']=df['AVERAGE_ACCT_AGE_MONTHS'].map(calc_months)
print(df[['AVERAGE_ACCT_AGE_MONTHS','AVERAGE_ACCT_AGE']].sample(n=10))


def convert_str_to_months(df, col_name):
    new_col = col_name + '_MONTHS' 
    df[new_col] = df[col_name].str.findall('\d+')
    df[new_col] = df[new_col].map(calc_months)


convert_str_to_months(df,'CREDIT_HISTORY_LENGTH')
print(df[['CREDIT_HISTORY_LENGTH_MONTHS','CREDIT_HISTORY_LENGTH']].sample(n=5))

df=df.drop(['AVERAGE_ACCT_AGE','CREDIT_HISTORY_LENGTH'],axis=1)

print(df['PERFORM_CNS_SCORE_DESCRIPTION'].value_counts())

df['LTV'] = df['LTV'].apply(lambda x: int(x.split('.')[0]) if not str(x).replace('.', '').isdigit() else x)
non_numeric_count = df['LTV'].apply(lambda x: not str(x).replace('.', '', 1).isdigit()).sum()

print(non_numeric_count)

df.to_csv('clean_data.csv')


