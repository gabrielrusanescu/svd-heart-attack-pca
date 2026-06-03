import numpy as np
import pandas as pd
import time

def load(f):
    datetabel = pd.read_csv(f)
    
    # se elimina id-ul pacientului (nu are relevanta aceasta coloana pentru calcul)
    datetabel.drop(columns=['Patient_ID'], errors='ignore', inplace=True)
    
    # transformam datele din cuvinte in cifre
    smoke_map = {'Never': 0, 'Former': 1, 'Current': 2}
    if 'smoking_status' in datetabel.columns:
        datetabel['smoking_status'] = datetabel['smoking_status'].map(smoke_map)
        
    fam_map = {'No': 0, 'Yes': 1}
    if 'family_history_heart_disease' in datetabel.columns:
        datetabel['family_history_heart_disease'] = datetabel['family_history_heart_disease'].map(fam_map)
    
    etichete = datetabel['risk_category'].values # ca sa stiu raspunsul corect (pe low este sanatos)
    trasaturimat = datetabel.drop(columns=['risk_category', 'heart_disease_risk_score'], errors='ignore')

    # eventuale lipsuri se inlocuiesc cu media coloanei
    trasaturimat.fillna(trasaturimat.mean(), inplace=True)
    
    x = trasaturimat.values.astype(np.float64)
        
    return x, etichete

if __name__ == "__main__":
    f = "cardiovascular_risk_dataset.csv"
    data, etichete = load(f)
    # print(np.shape(data))

    # selectam doar cazurile cu risc scazut (Low) pentru a defini normalitatea
    x_normal = data[etichete == 'Low']
    
    medie = np.mean(x_normal, axis=0)
    deviatiestandard = np.std(x_normal, axis=0)
    deviatiestandard[deviatiestandard == 0] = 1.0 
    x_normal = (x_normal - medie) / deviatiestandard
    
    start = time.perf_counter()
    print(f"Calcul SVD:")
    
    # matricea de covarianta
    C = (x_normal.T @ x_normal) / (np.shape(x_normal)[0] - 1)

    U, valproprii, Vt = np.linalg.svd(C)
    
    valsingulare = np.sqrt(np.maximum(valproprii, 0))
    valsingulare = np.sort(valsingulare)[::-1] 
    
    var = valsingulare**2 / np.sum(valsingulare**2)
    cum_var = np.cumsum(var)
    print(f"{'k':<10} | {'Pondere:':<10} | {'Cumulativ:':<10}")
    print("-" * 40)
    limit = min(14, np.shape(var)[0])
    for i in range(limit):
        print(f"{i+1:<10} | {var[i]*100:<10.2f}% | {cum_var[i]*100:<10.2f}%")
        
    end = time.perf_counter()
    print("-" * 40)
    print(f"Timp executie (varianta cu np.linalg): {end - start:.4f} secunde")