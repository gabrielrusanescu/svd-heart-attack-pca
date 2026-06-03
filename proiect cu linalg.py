import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

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
    print("\n")

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
    C = (Cstd.T @ Cstd) / (np.shape(Cstd)[0] - 1)

    U, S, Vt = np.linalg.svd(C)
    
    valproprii = S
    vectoriproprii = U
    vectoriproprii[:, 0] = vectoriproprii[:, 0] * -1
    
    Sigma_toate = np.sqrt(np.maximum(valproprii, 0))
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
    print(f"Timp executie (folosind np.linalg): {end - start:.4f} secunde")
    
    plotgrafic(Sigma_toate)
    plotpacienti(Cstd, etichete, vectoriproprii, valproprii)