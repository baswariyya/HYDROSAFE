import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
import joblib

#dataset loading
data = pd.read_csv("water_potability.csv")

#removing the null values with mean
data['ph']=data['ph'].fillna(data['ph'].mean())
data['Sulfate']=data['Sulfate'].fillna(data['Sulfate'].mean())
data['Trihalomethanes']=data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())

#assigning x and y values
x= data.drop("Potability",axis=1)
y= data["Potability"]

#preprocessing the data using StandardScalar to keep the values in the same scale
scaler =StandardScaler ()
x=scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

#to sort the imbalance in datset using smote to create new dataas synthetically
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

#RANDOM FOREST MODEL TRAINING
model_rf = RandomForestClassifier(n_estimators=800,max_depth=20, min_samples_split=4,min_samples_leaf=4,class_weight='balanced',random_state=42, n_jobs=-1  )
model_rf.fit(x_train_balanced, y_train_balanced)

#testing
proba_rf = model_rf.predict_proba(x_test)[:, 1] 
threshold = 0.52
pred_rf = (proba_rf >= threshold).astype(int)
print(np.column_stack((pred_rf, y_test)))

# Evaluation
print("Evaluation Results (RANDOM FOREST) ")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_rf))
print("Accuracy          : {:.2f}%".format(accuracy_score(y_test, pred_rf)*100))
print("ROC-AUC Score     : {:.3f}".format(roc_auc_score(y_test, proba_rf)))
print("\nClassification Report:\n", classification_report(y_test, pred_rf, digits=3))

#dumping
joblib.dump (model_rf,"water.pkl")
joblib.dump(scaler, "water_scaler.pkl")
print ("Training completed...!!!")