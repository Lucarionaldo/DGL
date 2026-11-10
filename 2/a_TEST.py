import numpy as np
from scipy.linalg import cholesky, lu, solve, solve_triangular

n_max = 13  # Zum Ausprobieren auf 14 oder 15 ändern
T = np.empty((n_max - 1, 6))

for zeile, n in enumerate(range(2, n_max + 1)):
    # Hilbertmatrix aufbauen. Python zählt bei 0 los.
    A = np.array([
        [1 / (i + j + 1) for j in range(n)]
        for i in range(n)
    ])

    # Erste und letzte Spalte addieren
    b = A[:, 0] + A[:, -1]

    # Exakte Lösung: vorne und hinten eine 1
    x = np.zeros(n)
    x[0] = 1
    x[-1] = 1
    x_norm = np.linalg.norm(x, np.inf)

    # 1. Direkt lösen
    x_bs = solve(A, b)
    fehler_bs = np.linalg.norm(x - x_bs, np.inf) / x_norm

    # 2. LR-Zerlegung mit Pivotisierung
    # SciPy liefert A = P @ L @ R. Daher verwenden wir P.T @ b.
    P, L, R = lu(A)
    y_lr = solve_triangular(L, P.T @ b, lower=True)
    x_lr = solve_triangular(R, y_lr)
    fehler_lr = np.linalg.norm(x - x_lr, np.inf) / x_norm

    # 3. Cholesky-Zerlegung: A = R_chol.T @ R_chol
    try:
        R_chol = cholesky(A)
        y_chol = solve_triangular(R_chol.T, b, lower=True)
        x_chol = solve_triangular(R_chol, y_chol)
        fehler_chol = np.linalg.norm(x - x_chol, np.inf) / x_norm
        probewert = np.linalg.norm(A @ x_chol - b, np.inf)
    except np.linalg.LinAlgError:
        print(f"Cholesky ist bei n = {n} fehlgeschlagen.")
        fehler_chol = np.nan
        probewert = np.nan

    kondition = np.linalg.cond(A, np.inf) * 1e-16
    T[zeile] = [n, kondition, fehler_bs, fehler_lr, fehler_chol, probewert]

print(" n   Kond. * 1e-16        Direkt            LR      Cholesky     Probewert")
for n, kondition, direkt, lr, chol, probe in T:
    print(
        f"{n:2.0f}   {kondition:13.4e}  {direkt:12.4e}  "
        f"{lr:12.4e}  {chol:12.4e}  {probe:12.4e}"
    )

# Vergleich für n = 2 bis 13 anhand des größten relativen Fehlers
verfahren = ["Direkt", "LR", "Cholesky"]
max_fehler = np.nanmax(T[:12, 2:5], axis=0)
schlechtestes = np.argmax(max_fehler)
print(f"\nGrößter relativer Fehler für n = 2 bis 13: {verfahren[schlechtestes]}")

# Bei n = 14 oder 15 können Rundungsfehler so stark ins Gewicht fallen,
# dass die Cholesky-Zerlegung scheitert. Dann steht dort nan.
# Das genaue Ergebnis hängt von der Rechenumgebung ab.