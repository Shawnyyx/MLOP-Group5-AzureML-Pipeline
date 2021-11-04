import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from azureml.core import Run


parser = argparse.ArgumentParser("train")


parser.add_argument("--train", type=str, help="train")
parser.add_argument("--test", type=str, help="test")
parser.add_argument("--model", type=str, help="model")


args = parser.parse_args()
run = Run.get_context()


run.log("Training start time", str(datetime.datetime.now()))


train=np.loadtxt(args.train+"/train.txt",dtype=float)
test=np.loadtxt(args.test+"/test.txt",dtype=float)


X_train = train[:,:-1]
Y_train = train[:,-1]
X_test = test[:,:-1]
Y_test = test[:,-1]


model = RandomForestClassifier()
model.fit(X_train, Y_train)


if not os.path.isdir(args.model):os.mkdir(args.model)
joblib.dump(model,args.model+"/rf_model.pkl")


result = model.score(X_test, Y_test)
run.log('Accuracy', result)


y_preds = model.predict_proba(X_test)
preds = y_preds[:,1]


from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(Y_test, preds)
auc_score = metrics.auc(fpr, tpr)


run.log("Training end time", str(datetime.datetime.now()))

run.log('AUC Score',auc_score)

for i in range(len(fpr)): run.log_row(name='ROC curve', False_Positive_Rate=fpr[i], True_Positive_Rate=tpr[i])
for i in range(len(fpr)):run.log_row(name='False Positive Rate', False_Positive_Rate=fpr[i])
for i in range(len(fpr)):run.log_row(name='True Positive Rate', True_Positive_Rate=tpr[i])

run.complete()