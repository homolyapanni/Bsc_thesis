from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

train_features=np.load('matrix_with_id_train.npy')
train_labels=np.load('labels_array_train.npy')

id = train_features[:,0]
train_features=train_features[:,1:]

model = LogisticRegression()
print("Fitting")
model.fit(train_features, train_labels)
print("Save the model")
save = 'model.sav'
joblib.dump(model, save)
