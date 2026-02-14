import cupy as cp

def calculate_fitness_batch(lines_tensor):
    """
    Calculates the number of valid Kobon triangles for a batch of line configurations.
    
    Args:
        lines_tensor (cp.ndarray): Shape (Batch, N, 2) containing (theta, rho).
                                   Must be float64.
                                   
    Returns:
        cp.ndarray: Shape (Batch,) containing integer count of valid triangles.
    """
    batch_size, n, _ = lines_tensor.shape
    epsilon = 1e-9 # Hardened tolerance
    
    # 1. Expand for Pairwise Intersections
    # Shape: (Batch, N, 1, 2) and (Batch, 1, N, 2)
    lines_i = lines_tensor[:, :, cp.newaxis, :]
    lines_j = lines_tensor[:, cp.newaxis, :, :]
    
    th_i = lines_i[..., 0]
    rho_i = lines_i[..., 1]
    th_j = lines_j[..., 0]
    rho_j = lines_j[..., 1]
    
    # Determinant: sin(th_j - th_i)
    det = cp.sin(th_j - th_i)
    
    # Parallel check (avoid div by zero)
    # We set invalid dets to NaN so x/y become NaN
    det_safe = cp.where(cp.abs(det) < epsilon, cp.nan, det)
    
    # Cramer's Rule for Intersection P_ij
    # x = (rho_i sin(th_j) - rho_j sin(th_i)) / det
    # y = (rho_j cos(th_i) - rho_i cos(th_j)) / det
    
    x_ij = (rho_i * cp.sin(th_j) - rho_j * cp.sin(th_i)) / det_safe
    y_ij = (rho_j * cp.cos(th_i) - rho_i * cp.cos(th_j)) / det_safe
    
    # Shape of intersections: (Batch, N, N, 2)
    # P_ij is the point where line i and line j meet.
    # Note: P_ii is NaN. P_ij == P_ji.
    
    # 2. Generate Triplet Combinations
    # We need to check all triplets (i, j, k).
    # Since N is small (14), we can pre-calculate indices on CPU or just use loops mainly?
    # No, we need vectorization. 
    # We can form a tensor of triplets. 
    # Triplet (i, j, k) implies Vertices: V1=P_ij, V2=P_jk, V3=P_ki.
    
    # Generate indices for combinations(N, 3)
    # Since N=14, C(14,3) = 364. This is small.
    # We can hardcode/generate the index array.
    
    import itertools
    triplet_indices = list(itertools.combinations(range(n), 3))
    num_triplets = len(triplet_indices)
    
    # Move indices to GPU
    # shape (NumTriplets, 3)
    triplets_gpu = cp.array(triplet_indices, dtype=cp.int32)
    
    idx_i = triplets_gpu[:, 0] # (NumTriplets,)
    idx_j = triplets_gpu[:, 1]
    idx_k = triplets_gpu[:, 2]
    
    # Extract Vertices for all triplets
    # We rely on fancy indexing into x_ij, y_ij (Batch, N, N)
    # We need (Batch, NumTriplets) vertices.
    # V1 = Intersection(i, j)
    # V2 = Intersection(j, k)
    # V3 = Intersection(k, i)
    
    # We expand x_ij to allow triplet indexing
    # We want V1_x shape: (Batch, NumTriplets)
    
    # Helper to gather vertices:
    # x_ij has shape (Batch, N, N). We pick (:, i, j)
    # This requires broadcasting the indices to fit batch? 
    # Or simplified:
    # x_ij_flat = x_ij.reshape(batch_size, -1)
    # flat_idx_ij = idx_i * n + idx_j
    # V1_x = x_ij_flat[:, flat_idx_ij]
    
    x_flat = x_ij.reshape(batch_size, n * n)
    y_flat = y_ij.reshape(batch_size, n * n)
    
    # Compute flat indices for the 3 points
    # Symmetric check: we computed all pairs, but i<j logic is consistent with combinations?
    # combinations(N,3) returns sorted i<j<k.
    # So pairs are (i,j), (j,k), (i,k). All satisfy index1 < index2 if we order them rights?
    # P_ij -> i,j (ok)
    # P_jk -> j,k (ok)
    # P_ki -> i,k (ok, since i < k)
    
    flat_idx_ij = idx_i * n + idx_j
    flat_idx_jk = idx_j * n + idx_k
    flat_idx_ik = idx_i * n + idx_k # intersection of i and k
    
    # Gather coordinates
    # v1: P_ij
    v1_x = x_flat[:, flat_idx_ij] # (Batch, NumTriplets)
    v1_y = y_flat[:, flat_idx_ij]
    
    # v2: P_jk
    v2_x = x_flat[:, flat_idx_jk]
    v2_y = y_flat[:, flat_idx_jk]
    
    # v3: P_ik (P_ki)
    v3_x = x_flat[:, flat_idx_ik]
    v3_y = y_flat[:, flat_idx_ik]
    
    # Check for NaN (invalid triangle due to parallel lines)
    # If any vertex is NaN, triangle is invalid.
    valid_geometry = ~(cp.isnan(v1_x) | cp.isnan(v2_x) | cp.isnan(v3_x))
    
    # 3. SEGMENT INTERSECTION TEST (Robustness)
    # A triangle is valid IFF NO other line L_m intersects any of its segments [V1,V2], [V2,V3], [V3,V1].
    
    # We need to test against ALL lines m.
    # Lines shape: (Batch, N, 2)
    # Vertices shape: (Batch, NumTriplets)
    
    # Expand Vertices -> (Batch, NumTriplets, 1)
    # Expand Lines -> (Batch, 1, N) (ignoring rho/theta components for now)
    
    # Let's perform the test for Segment [V1, V2] against Line m.
    # Segment defined by P1(v1_x, v1_y) and P2(v2_x, v2_y).
    # Line m defined by th_m, rho_m.
    
    # Signed Distance of point P(x,y) to line m: D = x*cos(th) + y*sin(th) - rho
    
    # We need cos_m, sin_m, rho_m tailored for broadcasting.
    # Lines: (Batch, N). We add axis 1 -> (Batch, 1, N)
    th_m = lines_tensor[:, :, 0][:, cp.newaxis, :] # (Batch, 1, N)
    rho_m = lines_tensor[:, :, 1][:, cp.newaxis, :] # (Batch, 1, N)
    
    cos_m = cp.cos(th_m)
    sin_m = cp.sin(th_m)
    
    # Expand Vertices for broadcasting against N lines
    # (Batch, NumTriplets, 1)
    v1_x_exp = v1_x[:, :, cp.newaxis]
    v1_y_exp = v1_y[:, :, cp.newaxis]
    v2_x_exp = v2_x[:, :, cp.newaxis]
    v2_y_exp = v2_y[:, :, cp.newaxis]
    v3_x_exp = v3_x[:, :, cp.newaxis]
    v3_y_exp = v3_y[:, :, cp.newaxis]
    
    # Calculate Distances of all 3 vertices to all N lines
    # Shape: (Batch, NumTriplets, N)
    
    d1 = v1_x_exp * cos_m + v1_y_exp * sin_m - rho_m
    d2 = v2_x_exp * cos_m + v2_y_exp * sin_m - rho_m
    d3 = v3_x_exp * cos_m + v3_y_exp * sin_m - rho_m
    
    # Check Intersection on Edges
    # Edge 1: [V1, V2]. Intersects line m if d1 * d2 < -epsilon
    # Edge 2: [V2, V3]. Intersects line m if d2 * d3 < -epsilon
    # Edge 3: [V3, V1]. Intersects line m if d3 * d1 < -epsilon
    
    # Strict inequality < -epsilon ensures we ignore cases where line passes THROUGH a vertex 
    # (distance ~ 0). The lines forming the triangle (i,j,k) will pass through vertices, 
    # so their products will be 0 (or > -eps). This correctly ignores self-intersections.
    
    has_cut_1 = (d1 * d2) < -epsilon
    has_cut_2 = (d2 * d3) < -epsilon
    has_cut_3 = (d3 * d1) < -epsilon
    
    # A line m cuts the triangle if it cuts ANY edge
    # Shape: (Batch, NumTriplets, N)
    cuts_any_edge = has_cut_1 | has_cut_2 | has_cut_3
    
    # A triangle is invalid if ANY line m cuts it.
    # Sum/Any over the N axis.
    # Shape: (Batch, NumTriplets)
    is_invalid = cp.any(cuts_any_edge, axis=2)
    
    # Final Validity
    # Must be geometrically valid (not parallel/NaN) AND have no extra cuts
    is_valid_triangle = valid_geometry & (~is_invalid)
    
    # Count valid triangles per batch item
    scores = cp.sum(is_valid_triangle, axis=1)
    
    return scores
