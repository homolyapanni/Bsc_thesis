import joblib
import numpy as np
import os
from sklearn import metrics

def read_label(path,filename):
    result = []
    read = os.listdir(path)
    for fn in read:
        if fn == filename:
            with open(fn, 'r', encoding="utf-8") as file:
                for row in file.readlines():
                    tpl = row.split(" ")
                    result.append((int(tpl[0]), float(tpl[1].strip("\n"))))
                    
    return result

def values(row_id,prediction,val_labels):
    inc=[]
    corr=[]
    for i,p,v in zip(row_id,prediction,val_labels):
        if p != v:
            r=[i,p,v]
            inc.append(r)
        if p == v:
            r=[i,p,v]
            corr.append(r)
            
    return inc,corr

        
path=os.getcwd()

val_features=np.load('matrix_with_id_val.npy')
val_labels=np.load('labels_array_val.npy')

model_name="model.sav"

id = val_features[:,0]
val_features=val_features[:,1:]


print(val_labels)

loaded_model = joblib.load(model_name)

print("Prediction")
predictions = loaded_model.predict(val_features)

print("Evaluation")
TP,TN,FP,FN = 0,0,0,0

for i,j in zip(val_labels,predictions):
  if i == 0 and j == 0:
    TP+=1
  if i == 0 and j == 1:
    FN+=1
  if i == 1 and j == 0:
    FP+=1
  if i == 1 and j == 1:
    TN+=1
    
print(TP,TN,FP,FN)
print("precision")
p=TP/(TP+FP)
print(p)
print("recall")
r=TP/(TP+FN)
print(r)
print("f-score")
f=(2*p*r)/(p+r)
print(f)
print("accuracy")
a=(TP+TN)/(TP+TN+FP+FN)
print(a)

print(loaded_model.coef_[0][0])

"""
print("Incorrect predictions")
incorrect,correct = values(id,predictions,val_labels)

with open('incorrect_gaz.txt', 'w') as file:
    file.write('\n'.join('{} {} {}'.format(i,p,r) for [i,p,r] in incorrect))
    
    
with open('correct_gaz.txt', 'w') as file:
    file.write('\n'.join('{} {} {}'.format(i,p,r) for [i,p,r] in correct))

"""
