import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('clean_data.csv',sep=',',index_col='UNIQUEID')

print(df.info())

print(df.nunique())

print(df['MOBILENO_AVL_FLAG'].value_counts())
df=df.drop(['MOBILENO_AVL_FLAG'], axis=1)

print(df[['SUPPLIER_ID','CURRENT_PINCODE_ID','EMPLOYEE_CODE_ID','BRANCH_ID','STATE_ID','MANUFACTURER_ID']].sample(10))
df=df.drop(['SUPPLIER_ID','CURRENT_PINCODE_ID','EMPLOYEE_CODE_ID','BRANCH_ID'], axis=1)


print(df['MANUFACTURER_ID'].value_counts(normalize=True))
sns.countplot(x='MANUFACTURER_ID',data=df)
plt.show()



print(df.groupby('MANUFACTURER_ID')['LOAN_DEFAULT'].value_counts(normalize=True).unstack(level=-1))

sns.catplot(data=df,kind='count',x='MANUFACTURER_ID',hue='LOAN_DEFAULT')
plt.show()


def explore_categorical(col_name):   
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

    sns.catplot(data=df,kind='count',x=col_name,hue='LOAN_DEFAULT')
    plt.show()


print(explore_categorical('DISBURSAL_MONTH'))


print(df['AGE'].describe())
sns.boxplot(x='AGE',data=df)
plt.show()

sns.displot(df['AGE'], kind='hist', kde=False)
plt.show()


print(df.groupby('LOAN_DEFAULT')['AGE'].describe())

sns.boxplot(x='AGE',y='LOAN_DEFAULT',data=df,orient='h')
plt.show()


def explore_continuous(col_name):
  
    print("{0} Summary".format(col_name))
    print("\n")
    print(df[col_name].describe())
    print("\n")

  
    sns.boxplot(x=col_name, data=df)
    plt.show()

 
    sns.displot(df[col_name],  kind='hist', kde=False)
    plt.show()

   
    print("{0} Grouped Summary".format(col_name))
    print("\n")
    print(df.groupby('LOAN_DEFAULT')[col_name].describe())

    
    sns.boxplot(x=col_name, y='LOAN_DEFAULT', data=df, orient="h")
    plt.show()


explore_continuous('DISBURSED_AMOUNT')

explore_categorical('AADHAR_FLAG')

df.to_csv('analysis.csv')


