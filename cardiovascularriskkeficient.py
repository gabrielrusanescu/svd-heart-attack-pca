import numpy as np
import pandas as pd
import time

def normavector(v):
    return np.sqrt(np.sum(v**2))

def fact_QR_HH(A):
    m, n = np.shape(A)
    R = np.copy(A).astype(float)
    Q = np.eye(m)
    
    for k in range(n):
        x = R[k:, k].reshape(-1, 1)
        normax = normavector(x)
        if normax < 1e-10:
            continue
        if x[0, 0] >= 0:
            sign = 1
        else:
            sign = -1
        alfa = -sign * normax
        e0 = np.zeros_like(x)
        e0[0, 0] = 1
        v = x - alfa * e0
        
        if normavector(v) < 1e-10:
            continue
        numitor = (v.T @ v).item()
        if numitor < 1e-10:
            continue
        H_mic = np.eye(m - k) - 2 * (v @ v.T) / numitor
        Hk = np.eye(m)
        Hk[k:, k:] = H_mic
        
        R = Hk @ R
        Q = Q @ Hk
        
    return Q, R

def Tridiag_Householder(A):
    n = np.shape(A)[0]
    T = np.copy(A).astype(float)
    Q = np.eye(n)

    for k in range(n - 2):
        x = T[k + 1:, k].reshape(-1, 1)
        normax = normavector(x)
        
        if normax < 1e-10:
            continue
            
        if x[0, 0] >= 0:
            sign = 1
        else:
            sign = -1
        alfa = -sign * normax
        
        e1 = np.zeros_like(x)
        e1[0, 0] = 1
        
        v = x - alfa * e1
        
        if normavector(v) < 1e-10:
            continue
        
        numitor = (v.T @ v).item()
        if numitor < 1e-10:
            continue
        H_mic = np.eye(n - k - 1) - 2 * (v @ v.T) / numitor
        
        H = np.eye(n)
        H[k + 1:, k + 1:] = H_mic
        
        T = H @ T @ H
        Q = Q @ H
        # fortez zerourile unde stiu sigur matematic ca trebuie sa fie 0
        T[k + 2:, k] = 0
        T[k, k + 2:] = 0
        
    return Q, T

def QR_iteration(A, TOL=1e-10):
    n = np.shape(A)[0]
    Q, T = Tridiag_Householder(A)
    V = Q
    maxiteratii = 10000
    for i in range(maxiteratii):
        elementeextdiag = np.abs(T - np.diag(np.diag(T)))
        maxelement = np.max(elementeextdiag)
        if maxelement < TOL:
            break
        Qextra, Rextra = fact_QR_HH(T)
        T = Rextra @ Qextra
        V = V @ Qextra
    return T

def load(f):
    datetabel = pd.read_csv(f)
    
    #se elimina id-ul pacientului (nu are relevanta aceasta coloana pentru calcul)
    datetabel.drop(columns=['Patient_ID'], errors='ignore', inplace=True)
    
    #transformam datele din cuvinte in cifre
    smoke_map = {'Never': 0, 'Former': 1, 'Current': 2}
    if 'smoking_status' in datetabel.columns:
        datetabel['smoking_status'] = datetabel['smoking_status'].map(smoke_map)
        
    fam_map = {'No': 0, 'Yes': 1}
    if 'family_history_heart_disease' in datetabel.columns:
        datetabel['family_history_heart_disease'] = datetabel['family_history_heart_disease'].map(fam_map)
    
    etichete = datetabel['risk_category'].values #ca sa stiu raspunsul corect (pe low este sanatos)
    trasaturimat = datetabel.drop(columns=['risk_category', 'heart_disease_risk_score'], errors='ignore')

    # eventuale lipsuri se inlocuiesc cu media coloanei
    trasaturimat.fillna(trasaturimat.mean(), inplace=True)
    
    x = trasaturimat.values.astype(np.float64)
        
    return x, etichete

if __name__ == "__main__":
    f = "cardiovascular_risk_dataset.csv"
    data, etichete = load(f)
    #print(np.shape(data))

    # selectam doar cazurile cu risc scazut (Low) pentru a defini normalitatea
    x_normal = data[etichete == 'Low']
    
    medie = np.mean(x_normal, axis=0)
    deviatiestandard = np.std(x_normal, axis=0)
    deviatiestandard[deviatiestandard == 0] = 1.0 
    x_normal = (x_normal - medie) / deviatiestandard
    
    start = time.perf_counter()
    print(f"Calcul SVD:")
    
    C = (x_normal.T @ x_normal) / (np.shape(x_normal)[0] - 1)

    valproprii = np.diag(QR_iteration(C))
    valsingulare = np.sqrt(np.maximum(valproprii, 0))
    valsingulare = np.sort(valsingulare)[::-1] 
    
    var = valsingulare**2 / np.sum(valsingulare**2)
    cum_var = np.cumsum(var)
    print(f"{'k':<10} | {'Pondere:':<10} | {'Cumulativ:':<10}")
    print("-"*40)
    limit = min(14, np.shape(var)[0])
    for i in range(limit):
        print(f"{i+1:<10} | {var[i]*100:<10.2f}% | {cum_var[i]*100:<10.2f}%")
        
    end = time.perf_counter()
    print("-"*40)
    print(f"Timp executie (varianta fara np.linalg): {end - start:.4f} secunde")