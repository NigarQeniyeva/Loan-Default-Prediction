import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

df = pd.read_csv('feature.csv', sep=',', index_col='UNIQUEID')

category_cols = ['MANUFACTURER_ID', 'STATE_ID', 'DISBURSAL_MONTH', 'DISBURSED_CAT', 'PERFORM_CNS_SCORE_DESCRIPTION', 'EMPLOYMENT_TYPE']
df[category_cols] = df[category_cols].astype('category')

small_cols = ['STATE_ID', 'LTV', 'DISBURSED_CAT', 'PERFORM_CNS_SCORE', 'DISBURSAL_MONTH', 'LOAN_DEFAULT']
loan_df_sml = df[small_cols]

data_dumm = pd.get_dummies(loan_df_sml, prefix_sep='_', drop_first=True)

x = data_dumm.drop(['LOAN_DEFAULT'], axis=1)
y = data_dumm['LOAN_DEFAULT']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(x_train, y_train)

print(logistic_model.score(x_test, y_test))

preds = logistic_model.predict(x_test)

conf_mat = confusion_matrix(y_test, preds)
print(conf_mat)

conf_mat_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
conf_mat_display.plot()
plt.show()

precision = precision_score(y_test, preds)
print(precision)

recall = recall_score(y_test, preds)
print(recall)

f1 = f1_score(y_test, preds)
print(f1)

probs = logistic_model.predict_proba(x_test)
fpr, tpr, threshold = roc_curve(y_test, probs[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

plot_roc_curve(fpr, tpr, roc_auc)

def eval_model(model, x_test, y_test):
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)

    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    roc_auc = auc(fpr, tpr)
    print("AUC: ", roc_auc)

    plot_roc_curve(fpr, tpr, roc_auc)

    results_df = pd.DataFrame()
    results_df['true_class'] = y_test
    results_df['predicted_class'] = preds
    results_df['default_prob'] = probs[:, 1]

    sns.kdeplot(data=results_df[results_df['true_class'] == 0]['default_prob'], label="No Default")
    sns.kdeplot(data=results_df[results_df['true_class'] == 1]['default_prob'], label="Default")
    plt.legend()
    plt.show()

    print(results_df.groupby('true_class')['predicted_class'].value_counts(normalize=True))

eval_model(logistic_model, x_test, y_test)
