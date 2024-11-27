import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import ConfusionMatrixDisplay,classification_report

data = pd.read_csv('C:/Users/pc/Documents/ML/logistic regression/cancer prediction project/The_Cancer_data_1500_V2.csv')
# print(data.head())

data_x = data.iloc[:,0:-1].values
data_y = data.iloc[:,-1].values
print(data_y)

X_train,X_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)


AB_clf = AdaBoostClassifier( algorithm = 'SAMME',random_state = 42)

AB_clf.fit(X_train, y_train)
AB_predictions = AB_clf.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, AB_predictions,
                                        labels = AB_clf.classes_, cmap = 'Blues')

plt.show()
print(classification_report(y_test, AB_predictions))
#print("Train Score:", reg.score(X_train,y_train))
#print("Test Score:", reg.score(X_test,y_test))

pickle.dump(AB_clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[58, 1, 16.085313321370478, 0,1,8.146250560259173,4.148219026764642,1]]))
