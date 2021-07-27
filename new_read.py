import csv
import os
import math
import random
import preprocessor as p
from tqdm import tqdm

def read_prep_split(path,filename,N):
    dataset = []
    i=0
    read = os.listdir(path)
    for fn in read:
        if fn == filename:
            with open(fn, 'r', encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in tqdm(reader):
                    label=0
                    if row[0] == 'id':
                        continue
                    if int(row[0]) == 346726:
                        continue
                    text = row[2]
                    text =p.clean(text)
                    text = text.lower()
                    
                    if float(row[1]) <= 0.3:
                        label = 0
                    
                    if 0.3 < float(row[1]) < 0.7:
                        continue
                    
                    if 0.7 <= float(row[1]):
                        label = 1
                        
                    new_row=[row[0],text,label]
                    dataset.append(new_row)
                    i += 1
                    if i == N:
                        break
  
    n=len(dataset)
    random.shuffle(dataset)

    
    train_length=math.floor(n*0.90)
   
    train_set=dataset[:train_length]
    val_set=dataset[train_length:]
    print("Saving")
    with open('train_set_bin_'+str(N)+'.csv','w',encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in train_set:
            csv_writer.writerow(i)
        
    with open('val_set_bin_'+str(N)+'.csv','w',encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in val_set:
            csv_writer.writerow(i)
       
    return train_set, val_set
    

path=os.getcwd()
filename="test_private_expanded.csv"
N=10
rd=read_prep_split(path,filename,N)
