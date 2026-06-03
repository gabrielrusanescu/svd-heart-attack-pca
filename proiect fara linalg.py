import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def normavector(v):
    return np.sqrt(np.sum(v**2))

def Tridiag_Householder(A):
    n = np.shape(A)[0]
    T = np.copy(A).astype(float)
    Q = np.eye(n)

    for k in range(n - 2):
        x = T[k + 1:, k].reshape(-1, 1)
        normax = np.linalg.norm(x)
        
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
        
        if np.linalg.norm(v) < 1e-10:
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

def fact_QR_HH(A):
    m, n = np.shape(A)
    R = np.copy(A).astype(float)
    Q = np.eye(m)
    
    for k in range(min(m, n)):
        x = R[k:, k].reshape(-1, 1)
        normax = np.linalg.norm(x)
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
        
        if np.linalg.norm(v) < 1e-10:
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
    return T, V

def SVD(A):
    m, n = np.shape(A)
    T, V = QR_iteration(A)
    valoriproprii = np.diag(T)
    valsingulare = np.sqrt(np.maximum(valoriproprii, 0))
    #valsingulare = np.maximum(valoriproprii, 0)
    
    indicisortare = np.argsort(valsingulare)[::-1]
    valsingulare, V = valsingulare[indicisortare], V[:, indicisortare]
    
    U = np.zeros((m, m))
    for i in range(min(m, n)):
        if valsingulare[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / valsingulare[i]
            
    for i in range(m):
        if np.linalg.norm(U[:, i]) < 1e-10:
            for k in range(m):
                v = np.zeros(m)
                v[k] = 1.0
                # Gram-Schmidt
                for j in range(i):
                    v -= np.dot(U[:, j], v) * U[:, j]
                if np.linalg.norm(v) > 1e-10:
                    U[:, i] = v / np.linalg.norm(v)
                    break
            
    S = np.zeros((m, n))
    for i in range(min(m, n)):
        S[i, i] = valsingulare[i]
        
    return U, S, V

def load(f):
    datetabel = pd.read_csv(f)
    
    # se elimina id-ul pacientului
    datetabel.drop(columns=['Patient_ID'], errors='ignore', inplace=True)
    
    # transformam datele din cuvinte in cifre
    smoke_map = {'Never': 0, 'Former': 1, 'Current': 2}
    if 'smoking_status' in datetabel.columns:
        datetabel['smoking_status'] = datetabel['smoking_status'].map(smoke_map)
        
    fam_map = {'No': 0, 'Yes': 1}
    if 'family_history_heart_disease' in datetabel.columns:
        datetabel['family_history_heart_disease'] = datetabel['family_history_heart_disease'].map(fam_map)
    
    etichete = datetabel['risk_category'].values
    trasaturimat = datetabel.drop(columns=['risk_category', 'heart_disease_risk_score'], errors='ignore')

    # eventuale lipsuri se inlocuiesc cu media coloanei
    trasaturimat.fillna(trasaturimat.mean(), inplace=True)
    
    x = trasaturimat.values.astype(np.float64)
        
    return x, etichete, trasaturimat

def afisponderi(vectoriproprii, coloane, k_max=3):
    print("\nAnaliza Ponderilor pentru primele k componente")
    
    for i in range(k_max): # se parcurg primele k componente
        print(f"\nComponenta Principala {i+1} (corespunde lui sigma_{i+1}):")
        
        pondere = vectoriproprii[:, i] #iau vectorul corespunzator componentei i
        modulpondere = np.abs(pondere)
        indicidesortare = np.argsort(modulpondere)[::-1]
        for j in indicidesortare[:3]:
            print(f"{coloane[j]}: {modulpondere[j]:.4f}")

def plotgrafic(sigma):
    #plt.figure(figsize=(8, 5))
    var = sigma**2 / np.sum(sigma**2)
    cum_var = np.cumsum(var) * 100
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.plot(range(1, len(sigma) + 1), sigma, 'o-', linewidth=2, color='green', label='Valori Singulare')
    ax1.set_xlabel('Componenta Principala (k)')
    ax1.set_ylabel('Valoare Singulara')
    
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cum_var) + 1), cum_var, 'o--', linewidth=2, color='blue', label='Varianta Cumulata (%)')
    ax2.set_ylabel('Varianta Cumulata (%)')
    
    plt.title('Importanta Componentelor Principale')
    ax1.grid(True, linestyle='--')
    plt.show()

def plotpacienti(data, etichete, vectoriproprii, valproprii):
    total_pacienti = len(etichete)
    pacienti_ok = np.sum(etichete == 'Low')
    pacienti_risc = total_pacienti - pacienti_ok
    
    print("\n")
    print("Statistici")
    print(f"Total pacienti: {total_pacienti}")
    print(f"Pacienti OK (Low): {pacienti_ok}")
    print(f"Pacienti cu risc (Medium/High): {pacienti_risc}")

    #identific variabilele dominante
    idx1 = np.argmax(np.abs(vectoriproprii[:, 0]))
    idx2 = np.argmax(np.abs(vectoriproprii[:, 1]))
    
    V2 = vectoriproprii[:, :2]
    scoruri = data @ V2
    procente = (valproprii / np.sum(valproprii)) * 100
    
    plt.figure(figsize=(10, 7))
    sanatosi = (etichete == 'Low')
    
    # Plotare puncte
    plt.scatter(scoruri[sanatosi, 0], scoruri[sanatosi, 1], c='green', edgecolors='black', linewidth=0.5, s=65, label='OK (Risc Scazut)')
    plt.scatter(scoruri[~sanatosi, 0], scoruri[~sanatosi, 1], c='red', edgecolors='black', linewidth=0.5, s=65, label='Atentie (Risc Mediu/Mare)')
    
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    
    plt.xlabel(f'Componenta principala 1 ({procente[0]:.1f}%)', fontweight='bold')
    plt.ylabel(f'Componenta principala 2 ({procente[1]:.1f}%)', fontweight='bold')
    plt.title('Riscul de infarct', fontsize=14, fontweight='bold')
    
    plt.grid(True, linestyle='--')
    plt.show()
    
if __name__ == "__main__":
    f = "cardiovascular_risk_dataset.csv"
    data, etichete, trasaturimat = load(f)
    
    medie = np.mean(data, axis=0)
    deviatiestandard = np.std(data, axis=0)
    deviatiestandard[deviatiestandard == 0] = 1.0 
    Cstd = (data - medie) / deviatiestandard
    
    start = time.perf_counter()
    C = (Cstd.T @ Cstd) / (np.shape(Cstd)[0]
    #U, S, Vt = np.linalg.svd(C)
    U, S, Vt = SVD(C)
    Vt = Vt.T
    
    for i in range(np.shape(U)[1]):
        norma = np.linalg.norm(U[:, i])
        if norma > 1e-10:
            U[:, i] /= norma
    #de aici este nou
    """
    for i in range(np.shape(U)[1]):
        maxi = np.argmax(np.abs(U[:, i]))
        if U[maxi, i] < 0:
            U[:, i] *= -1
            Vt[i, :] *= -1
    """
    S = np.diag(S)
    S = S**2
    #pana aici
    valproprii = S
    vectoriproprii = U
    #vectoriproprii[:, 0] = vectoriproprii[:, 0] * -1
    Sigma_toate = np.sqrt(np.maximum(valproprii, 0))
    #Sigma_toate = valproprii
    Vt_toate = vectoriproprii.T
    print("\nRezultate Aproximare Rang k:")
    print(f"{'k':<20} | {'Sigma_k':<20} | {'Norma 2 (C - Ck)':<20} | {'Norma Frobenius (C - Ck)':<20}")
    print("-" * 80)
    limit = min(14, len(Sigma_toate))
    
    for k in range(1, limit + 1):
        # selectez primele k componente
        Vk = Vt_toate[:k, :].T  
        
        Ck = C @ Vk @ Vk.T
        
        reziduu = C - Ck
        
        norma_frob = np.linalg.norm(reziduu, 'fro')
        norma_2 = np.linalg.norm(reziduu, 2)
        
        sigma_k = Sigma_toate[k-1]
        
        print(f"{k:<20} | {sigma_k:<20.4f} | {norma_2:<20.4f} | {norma_frob:<20.4f}")
    
    print("-" * 80)
    coloane = trasaturimat.columns
    afisponderi(vectoriproprii, coloane, k_max=3)
    end = time.perf_counter()
    print(f"\nTimp executie (fara np.linalg): {end - start:.4f} secunde")
    
    plotgrafic(Sigma_toate)
    plotpacienti(Cstd, etichete, vectoriproprii, valproprii)