import numpy as np
import json
import glob
import os

def convert_all():
    npy_files = glob.glob("*.npy")
    print(f"Found {len(npy_files)} .npy files.")
    
    for f in npy_files:
        try:
            genome = np.load(f)
            # Genome shape: (N, 2)
            
            # Extract info from filename if possible
            # best_N14_K54.npy
            parts = f.replace(".npy", "").split("_")
            n_val = "Unknown"
            k_val = "Unknown"
            
            for p in parts:
                if p.startswith("N"): n_val = p[1:]
                if p.startswith("K"): k_val = p[1:]
            
            # Build list of line dicts
            lines_data = []
            for i in range(genome.shape[0]):
                lines_data.append({
                    "id": i,
                    "theta": float(genome[i, 0]),
                    "rho": float(genome[i, 1])
                })
            
            out_data = {
                "n": int(n_val) if n_val.isdigit() else n_val,
                "k": int(k_val) if k_val.isdigit() else k_val,
                "lines": lines_data
            }
            
            json_name = f.replace(".npy", ".json")
            with open(json_name, "w") as jf:
                json.dump(out_data, jf, indent=4)
                
            print(f"Converted {f} -> {json_name}")
            
        except Exception as e:
            print(f"Error converting {f}: {e}")

if __name__ == "__main__":
    convert_all()
