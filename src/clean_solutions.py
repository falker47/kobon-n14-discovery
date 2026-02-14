import os
import json
import numpy as np
import hashlib

# CONFIGURA QUI LA CARTELLA
SOLUTIONS_DIR = "solutions"  # Assicurati che il path sia giusto

def get_geometric_fingerprint(lines_data):
    """
    Crea un hash univoco basato sulla geometria.
    Arrotonda a 4 decimali per ignorare le micro-vibrazioni.
    """
    # 1. Estrai array (N, 2)
    matrix = []
    for l in lines_data:
        matrix.append([l['theta'], l['rho']])
    arr = np.array(matrix)
    
    # 2. Ordina le linee (per gestire permutazioni diverse della stessa forma)
    # Ordina per Theta, poi per Rho
    ind = np.lexsort((arr[:,1], arr[:,0]))
    arr_sorted = arr[ind]
    
    # 3. Arrotonda (Tolleranza ~0.0001)
    arr_rounded = np.round(arr_sorted, 4)
    
    # 4. Hash bytes
    return hashlib.md5(arr_rounded.tobytes()).hexdigest()

def clean_duplicates():
    print(f"--- ANALISI CARTELLA: {SOLUTIONS_DIR} ---")
    files = [f for f in os.listdir(SOLUTIONS_DIR) if f.endswith('.json')]
    print(f"File trovati: {len(files)}")
    
    unique_hashes = {}
    duplicates = []
    
    for filename in files:
        filepath = os.path.join(SOLUTIONS_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Calcola impronta
            fingerprint = get_geometric_fingerprint(data['lines'])
            
            if fingerprint in unique_hashes:
                duplicates.append(filepath)
            else:
                unique_hashes[fingerprint] = filepath
                
        except Exception as e:
            print(f"Errore lettura {filename}: {e}")

    print(f"\n--- RISULTATO ---")
    print(f"Soluzioni UNICHE trovate: {len(unique_hashes)}")
    print(f"Duplicati (Cloni): {len(duplicates)}")
    
    if len(duplicates) > 0:
        confirm = input("Vuoi cancellare i duplicati? (y/n): ")
        if confirm.lower() == 'y':
            for d in duplicates:
                os.remove(d)
            print("Pulizia completata.")
        else:
            print("Operazione annullata.")
    else:
        print("Nessun duplicato da cancellare.")

if __name__ == "__main__":
    clean_duplicates()