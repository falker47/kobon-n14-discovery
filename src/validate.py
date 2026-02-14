import json
import numpy as np
import math

def verify_solution_robust(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    lines = data['lines']
    n = len(lines)
    # Converti in array numpy per calcoli vettoriali
    thetas = np.array([l['theta'] for l in lines])
    rhos = np.array([l['rho'] for l in lines])
    
    print(f"--- VERIFICA INDIPENDENTE N={n} ---")
    
    # 1. Calcolo Intersezioni (Punti x, y)
    intersections = []
    eps = 1e-9 # Tolleranza per parallelismo
    
    for i in range(n):
        for j in range(i + 1, n):
            det = np.sin(thetas[j] - thetas[i])
            if abs(det) < eps: continue # Parallele
            
            # Cramer
            x = (rhos[i]*np.sin(thetas[j]) - rhos[j]*np.sin(thetas[i])) / det
            y = (rhos[j]*np.cos(thetas[i]) - rhos[i]*np.cos(thetas[j])) / det
            intersections.append({'lines': (i, j), 'pt': (x, y)})

    print(f"Punti di intersezione calcolati: {len(intersections)}")

    # 2. Ricerca Triangoli
    # Un triangolo è formato da 3 rette (i, j, k) che si intersecano a coppie
    triangles = []
    
    import itertools
    for i, j, k in itertools.combinations(range(n), 3):
        # Trova i 3 vertici coinvolti
        v_ij = next((p['pt'] for p in intersections if p['lines'] == (i, j)), None)
        v_jk = next((p['pt'] for p in intersections if p['lines'] == (j, k)), None)
        v_ki = next((p['pt'] for p in intersections if p['lines'] == (i, k)), None) # o (k, i) ordinato
        
        if not v_ij or not v_jk or not v_ki: continue
        
        # Calcolo Centroide (Baricentro)
        cx = (v_ij[0] + v_jk[0] + v_ki[0]) / 3.0
        cy = (v_ij[1] + v_jk[1] + v_ki[1]) / 3.0
        
        # 3. VERIFICA "VUOTO": Nessuna altra retta 'm' deve passare "vicino" al centroide?
        # No, il test rigoroso è: Il centroide sta dalla parte "giusta" di tutte le 3 rette?
        # E nessuna altra retta interseca i lati?
        # Metodo semplificato robusto: controlla se il baricentro è intersecato da altre rette?
        # No. Metodo Kobon Standard:
        # Un triangolo è valido se nessun'altra retta attraversa il suo perimetro.
        
        is_valid = True
        # Vertici del triangolo
        verts = [v_ij, v_jk, v_ki]
        
        for m in range(n):
            if m in (i, j, k): continue
            
            # Controlla se la retta m separa i vertici (cioè passa in mezzo)
            # Calcola distanza con segno per ogni vertice
            dists = []
            for vx, vy in verts:
                d = vx * np.cos(thetas[m]) + vy * np.sin(thetas[m]) - rhos[m]
                dists.append(d)
            
            # Se ci sono distanze positive E negative (e non zero), la retta taglia il triangolo
            d_min = min(dists)
            d_max = max(dists)
            
            if d_min < -1e-5 and d_max > 1e-5:
                is_valid = False
                break
        
        if is_valid:
            triangles.append((i, j, k))

    print(f"--- RISULTATO FINALE ---")
    print(f"Triangoli Rilevati: {len(triangles)}")
    if len(triangles) == 54:
        print("✅ SUCCESSO: 54 Triangoli CONFERMATI.")
    else:
        print(f"⚠️ ATTENZIONE: Rilevati {len(triangles)}. Possibile errore di precisione nel generatore.")

verify_solution_robust('best_N14_K54.json')