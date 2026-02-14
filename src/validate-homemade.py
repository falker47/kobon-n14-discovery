import math
from itertools import combinations

# DATI UTENTE
lines_data = [
    {"id": 0, "theta": 2.581973618355362, "rho": -1.4361014039205287},
    {"id": 1, "theta": 1.3244499516441954, "rho": 0.6217439887253811},
    {"id": 2, "theta": 1.5592384297106614, "rho": 1.6258255741602694},
    {"id": 3, "theta": 0.34226419677867326, "rho": 0.0857180542701795},
    {"id": 4, "theta": 0.5154721754131635, "rho": 1.9447522663536025},
    {"id": 5, "theta": 2.0148338011894853, "rho": 0.05191036854954184},
    {"id": 6, "theta": 2.371027487727583, "rho": -0.5559487124881608},
    {"id": 7, "theta": 2.6274869399441205, "rho": 1.976637470219313},
    {"id": 8, "theta": 0.7829290242720544, "rho": -0.6743848424733729},
    {"id": 9, "theta": 2.439838889318842, "rho": 0.8000410010285611},
    {"id": 10, "theta": 0.6782158618094248, "rho": -0.04135732565017096},
    {"id": 11, "theta": 1.614744960635221, "rho": 0.23101713181572722},
    {"id": 12, "theta": 2.4945517317639174, "rho": 0.20145266565179204},
    {"id": 13, "theta": 0.008553391847996746, "rho": 0.6584175665077822}
]

EPSILON = 1e-9

class Line:
    def __init__(self, uid, theta, rho):
        self.id = uid
        self.theta = theta
        self.rho = rho
        self.cos_t = math.cos(theta)
        self.sin_t = math.sin(theta)

    def dist(self, x, y):
        # Signed distance from point (x,y) to line
        return x * self.cos_t + y * self.sin_t - self.rho

def get_intersection(l1, l2):
    det = l1.cos_t * l2.sin_t - l1.sin_t * l2.cos_t
    if abs(det) < EPSILON:
        return None # Parallel
    x = (l1.rho * l2.sin_t - l2.rho * l1.sin_t) / det
    y = (l2.rho * l1.cos_t - l1.rho * l2.cos_t) / det
    return (x, y)

def verify_kobon(lines_data):
    lines = [Line(d['id'], d['theta'], d['rho']) for d in lines_data]
    n = len(lines)
    kobon_triangles = 0
    valid_triangles = []

    # Iterate all triplets
    for tri in combinations(lines, 3):
        l1, l2, l3 = tri
        
        # Calculate vertices
        v1 = get_intersection(l1, l2)
        v2 = get_intersection(l2, l3)
        v3 = get_intersection(l3, l1)

        if not v1 or not v2 or not v3:
            continue # Parallel lines involved

        # Check for degenerate triangle (vertices too close)
        if (math.hypot(v1[0]-v2[0], v1[1]-v2[1]) < EPSILON or
            math.hypot(v2[0]-v3[0], v2[1]-v3[1]) < EPSILON or
            math.hypot(v3[0]-v1[0], v3[1]-v1[1]) < EPSILON):
            continue

        # Kobon condition: No other line intersects the INTERIOR
        is_kobon = True
        for other in lines:
            if other.id in [l1.id, l2.id, l3.id]:
                continue
            
            # Evaluate signed distance of all 3 vertices to the other line
            d1 = other.dist(*v1)
            d2 = other.dist(*v2)
            d3 = other.dist(*v3)

            # If signs are mixed (strictly positive AND strictly negative), line cuts interior
            # We allow touching vertices (dist approx 0)
            
            has_pos = (d1 > EPSILON) or (d2 > EPSILON) or (d3 > EPSILON)
            has_neg = (d1 < -EPSILON) or (d2 < -EPSILON) or (d3 < -EPSILON)

            if has_pos and has_neg:
                is_kobon = False
                break
        
        if is_kobon:
            kobon_triangles += 1
            valid_triangles.append([l1.id, l2.id, l3.id])

    return kobon_triangles, valid_triangles

count, triangles = verify_kobon(lines_data)
print(f"Total Lines: {len(lines_data)}")
print(f"Kobon Triangles Found: {count}")
if count == 54:
    print("RESULT: VALID RECORD (Matches Claim)")
else:
    print(f"RESULT: MISMATCH (Claim: 54, Actual: {count})")