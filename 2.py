import numpy as np

# Befestigungspunkte der Federn
a1 = np.array([0.0, 0.0])
a2 = np.array([1.0, 1.0])
befestigungen = [a1, a2]

# Federkonstanten und Ruhelängen
s = [10.0, 10.0]
ell = [2.0, 2.0]

m = 1.0
g = 9.81


def F(x):
    #Gesamtkraft auf die Masse am Ort x
    kraft = np.array([0.0, -m * g])

    for a, si, ell_i in zip(befestigungen, s, ell):
        v = a - x
        d = np.linalg.norm(v)

        # Federkraft: Betrag s_i * (Federlänge - Ruhelänge)
        kraft += si * (d - ell_i) * v / d

    return kraft


def jacobi(x):
    #Jacobi-Matrix der Gesamtkraft
    J = np.zeros((2, 2))
    I = np.eye(2)

    for a, si, ell_i in zip(befestigungen, s, ell):
        v = a - x
        d = np.linalg.norm(v)

        J -= si * (
            (1 - ell_i / d) * I
            + ell_i / d**3 * np.outer(v, v)
        )

    return J


def newton(x, tol=1e-10, max_iter=100):
    #Newton-Verfahren für F(x) = 0
    x = np.array(x, dtype=float)

    for iteration in range(max_iter):
        if np.linalg.norm(F(x)) < tol:
            return x, iteration

        # J(x) * Schritt = F(x)
        schritt = np.linalg.solve(jacobi(x), F(x))
        x = x - schritt

    raise RuntimeError("Newton-Verfahren ist nicht konvergiert.")


startpunkte = [
    np.array([0.0, -4.0]),
    np.array([0.0, 4.0])
]

for x0 in startpunkte:
    gleichgewicht, schritte = newton(x0)

    print(f"Startpunkt: {x0}")
    print(f"Gleichgewicht: {gleichgewicht}")
    print(f"Newton-Schritte: {schritte}")
    print(f"Restkraft: {F(gleichgewicht)}")
    print()


#Wie sind die unterschiedlichen Ergebnisse der beiden Startpunkte zu erklären?

# Die Gleichung F_gesamt(x) = 0 hat mehrere Lösungen (Gleichgewichtslagen).
# Das Newton-Verfahren ist ein lokales Verfahren und konvergiert gegen die
# Nullstelle in der Nähe des Startpunkts.
# Startpunkt (0, -4): Gleichgewicht unterhalb der Aufhängungen, beide Federn
# sind gedehnt und halten die Masse gegen die Gewichtskraft (stabil).
# Startpunkt (0, 4): Gleichgewicht oberhalb der Aufhängungen, Feder 2 ist
# gestaucht und drückt die Masse nach oben, Feder 1 ist leicht gedehnt.