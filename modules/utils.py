import csv
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_MAIN_LINE_PATH = Path(__file__).resolve().parent.parent / "main_line_sol.csv"

def create_distance_matrix(df):
    coordinates = df[['x', 'y']].to_numpy(dtype=float)
    n = len(coordinates)
    
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n): #(only upper half due to symmetry)
            distance = np.linalg.norm(coordinates[i] - coordinates[j])
            
            distance_matrix[i,j] = distance
            distance_matrix[j,i] = distance  # Fill the symmetric entry
    city_names = df["City"].tolist()
    return pd.DataFrame(
        distance_matrix,
        index=city_names,
        columns=city_names,
    )





def write_solution(route, cable_length, fixed_connections, filename=DEFAULT_MAIN_LINE_PATH):
    with open(filename, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["Order", "City", "Within-area cable length", "Fixed connections length", "Total cable length"])
        for order, city in enumerate(route, start=1):
            writer.writerow([order, city, f"{cable_length:.2f}", f"{fixed_connections:.2f}",
                             f"{cable_length + fixed_connections:.2f}"])


## Tests ect
def check_symmetric_matrix(m : np.ndarray):
    """
    Check if a matrix is symmetric.
    """
    return np.allclose(m, m.T)
