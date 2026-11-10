import numpy as np
import matplotlib.pyplot as plt

# Eckpunkte der Ente
ente = np.array([
    [0, 1, 1, 1.5, 2.5, 3, 3, 4, 2, 2, 1, -3.7, -5, -4, -3, 0],
    [1, 0, 3.5, 4, 4, 3.5, 3, 3, 2, -1, -2, -2, 1, 0, 1, 1]
], dtype=float)

# Hilbertmatrix für n = 2
A = np.array([
    [1, 1 / 2],
    [1 / 2, 1 / 3]
], dtype=float)

U, singulaerwerte, VT = np.linalg.svd(A)
Sigma = np.diag(singulaerwerte)

# Punkte eines Kreises mit Radius 6 erzeugen
t = np.linspace(0, 2 * np.pi, 300)
radius = 6

kreis = np.array([
    radius * np.cos(t),
    radius * np.sin(t)
])

# Ente und Kreis nach den drei Schritten transformieren
ente_1 = ente
kreis_1 = kreis

ente_2 = VT @ ente_1
kreis_2 = VT @ kreis_1

ente_3 = Sigma @ ente_2
kreis_3 = Sigma @ kreis_2

ente_4 = U @ ente_3
kreis_4 = U @ kreis_3

# Daten für die vier Diagramme
bilder = [
    (ente_1, kreis_1, "Ausgangsbild"),
    (ente_2, kreis_2, r"nach orth. Trafo. $V^T$"),
    (ente_3, kreis_3, r"nach Streckung $\Sigma$"),
    (ente_4, kreis_4, r"nach orth. Trafo. $U$")
]

# Vier Diagramme nebeneinander zeichnen
fig, achsen = plt.subplots(1, 4, figsize=(16, 5))

for ax, (ente_bild, kreis_bild, titel) in zip(achsen, bilder):
    ax.plot(
        ente_bild[0, :],
        ente_bild[1, :],
        color="0.25",
        linewidth=3
    )

    ax.plot(
        kreis_bild[0, :],
        kreis_bild[1, :],
        color="0.25",
        linewidth=3
    )

    ax.set_title(titel, fontsize=14)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-10, 11, 1))
    ax.set_yticks(np.arange(-10, 11, 1))
    ax.grid(True, color="0.75", linewidth=0.6)

plt.tight_layout()
plt.show()