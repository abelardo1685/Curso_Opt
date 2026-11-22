# -*- coding: utf-8 -*-
"""OCA aplicada al problema de Fritz (2026-11-17, ARP: «una implementación de OCA a un problema; prueba con el caso de optimización
que hemos trabajado y evalúa si cumple»).

El caso es el de la sección 3 del cuaderno S4: agua contra riesgo con 200 geologías (ensamble con la zona del pozo 2). La OCA
(Rodríguez-Pretelín, Nowak y Morales-Casique, 2022) elige con OMOPSO K geologías representativas. En el artículo el subconjunto debe
reproducir los isopercentiles del mapa probabilístico de la zona de protección; aquí el «mapa» está en el espacio de decisiones:

    celda del mapa                      ->  una decisión de bombeo de referencia (malla de 0 a 200 m³/d cada 20: 1 331 decisiones)
    probabilidad de pertenecer al WHPA  ->  riesgo de la decisión (fracción de geologías en que rompe alguna regla)
    isopercentiles 10, 50 y 90          ->  las regiones de decisiones con riesgo <= 0, 10 y 20 %
    error de cada isopercentil          ->  decisiones de referencia mal clasificadas en cada región (tres objetivos de OMOPSO)

La prueba: con el subconjunto se optimiza (la decisión de más agua con riesgo <= 0, 10 y 20 % en la malla fina de 68 921 decisiones) y
la decisión se valida en las 200 geologías NUEVAS. Se compara con las 200 geologías, con las primeras K y con 300 subconjuntos al azar.

Uso (desde Session_4):  ../../.venv_curso/bin/python Información/codigo/oca_fritz.py      (unos 40 s; escribe Información/figuras/oca_fritz.png)
"""
import os
import sys
import time
import warnings

AQUI = os.path.dirname(os.path.abspath(__file__))
S4 = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path[:0] = [os.path.join(S4, ".."), AQUI, os.path.join(AQUI, "pyOMOPSO")]
warnings.filterwarnings("ignore")
import numpy as np                                             # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import hidra                                                   # noqa: E402
from ParOMOPSO_det import parOMOPSO_det as parOMOPSO           # noqa: E402

S_MAX, S_HUM, DEMANDA = 3.0, 1.25, 1200.0
ZONA = (735.0, 450.0, 300.0, -2.0)
NIVELES = (0.0, 0.10, 0.20)                                    # los tres niveles de riesgo: las tres «isolíneas» que la OCA reproduce
OMOPSO = {'MaxIter': 100, 'nPop': 40, 'w1': 0.1, 'w2': 0.5, 'e': 0.001, 'c1': 1.5, 'c2': 2, 'nGrid': 7, 'alpha': 0.1,
          'wdamp': 0.99, 'r1': 0, 'r2': 1, 'verbose': 0}


def malla(paso, tope):
    g = np.arange(0, tope + 1, paso, dtype=float)
    return np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T


def fallas(P, A, H):
    """(decisiones, geologías): ¿rompe la decisión alguna regla en esa geología?"""
    F = np.zeros((len(P), len(A)), dtype=bool)
    for k in range(len(A)):
        F[:, k] = ((P @ A[k].T) > S_MAX + 1e-9).any(axis=1) | ((P @ H[k]) > S_HUM + 1e-9)
    return F


def severidad(A, H, p=(100.0, 100.0, 100.0)):
    """Qué tan exigente es cada geología con un bombeo de prueba: la regla más cargada, como fracción de su límite."""
    p = np.asarray(p)
    return np.maximum((A @ p).max(axis=1) / S_MAX, (H @ p) / S_HUM)


def oca(F_ref, K, semilla=0, orden=None, parametros=OMOPSO):
    """Elige K geologías cuyas regiones de riesgo reproducen las del ensamble completo.

    F_ref: (decisiones de referencia, N geologías) de fallas. orden: las geologías ordenadas (por severidad) para que dos posiciones
    vecinas de la partícula sean geologías parecidas. Devuelve (índices elegidos, error en cada nivel)."""
    N = F_ref.shape[1]
    orden = np.arange(N) if orden is None else orden
    R_ref = F_ref.mean(axis=1)
    dentro = [R_ref <= t + 1e-12 for t in NIVELES]              # la región de cada nivel, con el ensamble completo

    def subconjunto(q):                                         # la posición de la partícula (K números de 0 a N) como geologías
        return orden[np.unique(np.minimum(np.floor(q).astype(int), N - 1))]

    def error(S):                                               # decisiones mal clasificadas en cada región
        r = F_ref[:, S].mean(axis=1)
        return np.array([np.sum((r <= t + 1e-12) != d) for t, d in zip(NIVELES, dentro)], dtype=float)

    def costo(q):
        S = subconjunto(q)
        if len(S) < K:                                          # geologías repetidas: castigo
            return np.full((3, 1), 1e4)
        return (error(S) + 1e-3)[:, None]

    np.random.seed(semilla)
    ef, rep, _ = parOMOPSO({'CostFunction': costo, 'nVar': K, 'VarMin': 0, 'VarMax': N - 1e-6}, dict(parametros))
    ef = ef if len(ef) else rep                                 # pyOMOPSO puede vaciar el archivo: entonces, el repositorio
    F = np.array([np.ravel(s['cost']) for s in ef]); X = np.array([np.ravel(s['position']) for s in ef])
    ok = F[:, 0] < 1e3; F, X = F[ok], X[ok]
    Fn = (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 1e-12)
    k = int(np.argmin(np.linalg.norm(Fn, axis=1)))             # la solución de compromiso: la más cercana al ideal
    S = np.sort(subconjunto(X[k]))
    return S, error(S)


def main():
    t0 = time.time()
    OPT = hidra.ensamble(200, "optimizacion", zona=ZONA, verbose=False); VAL = hidra.ensamble(200, "validacion", zona=ZONA, verbose=False)
    A, H = OPT["alpha"], OPT["alpha_h"]; N = len(A)
    MALLA = malla(10, 400); Z = MALLA.sum(axis=1)               # la malla fina del cuaderno: donde se optimiza
    FALLAS = fallas(MALLA, A, H); R_val = fallas(MALLA, VAL["alpha"], VAL["alpha_h"]).mean(axis=1)
    REF = malla(20, 200); F_ref = fallas(REF, A, H)             # la referencia de la OCA: 1 331 decisiones en la zona de bajo riesgo
    ORDEN = np.argsort(severidad(A, H))

    def optimizar(S):                                           # con las geologías S: para cada nivel, (agua, riesgo real en validación)
        rs = FALLAS[:, S].mean(axis=1); filas = []
        for r in NIVELES:
            ok = rs <= r + 1e-12; i = np.flatnonzero(ok)[np.argmax(Z[ok])]; filas.append((Z[i], R_val[i]))
        return np.array(filas)

    def curva(S):                                               # el frente agua–riesgo que promete S, y el riesgo real de cada punto
        rs = FALLAS[:, S].mean(axis=1); pts = []
        for r in np.unique(rs):
            ok = rs <= r + 1e-12; i = np.flatnonzero(ok)[np.argmax(Z[ok])]
            if not pts or Z[i] > pts[-1][0]:
                pts.append((Z[i], r, R_val[i]))
        return np.array(pts)

    completo = optimizar(np.arange(N))
    print(f"referencia de la OCA: {len(REF)} decisiones; en las regiones de riesgo <= 0/10/20 %: {[(F_ref.mean(1) <= t + 1e-12).sum() for t in NIVELES]}")
    print("\n                               agua [m³/d] pidiendo 0/10/20 %     riesgo REAL en 200 geologías nuevas")
    print(f"{'las 200 geologías':30s} {completo[:, 0].astype(int)}                {np.round(100 * completo[:, 1], 1)} %")
    rng = np.random.default_rng(0); guardar = {}
    for K in (10, 20):
        primeras = optimizar(np.arange(K))
        azar = np.array([optimizar(rng.choice(N, K, replace=False)) for _ in range(300)])
        t1 = time.time(); sel = [oca(F_ref, K, semilla=s, orden=ORDEN) for s in range(8)]; t_oca = (time.time() - t1) / 8
        res = np.array([optimizar(S) for S, _ in sel])
        d_azar = np.abs(azar[:, :, 1] - completo[:, 1]).sum(axis=1); d_oca = np.abs(res[:, :, 1] - completo[:, 1]).sum(axis=1)
        print(f"\nK = {K}")
        print(f"{'  las primeras ' + str(K):30s} {primeras[:, 0].astype(int)}                {np.round(100 * primeras[:, 1], 1)} %")
        print(f"{'  al azar (mediana de 300)':30s} {np.median(azar[:, :, 0], 0).astype(int)}                {np.round(100 * np.median(azar[:, :, 1], 0), 1)} %"
              f"   (del 10 al 90 %: {np.round(100 * np.percentile(azar[:, :, 1], 10, 0), 1)} a {np.round(100 * np.percentile(azar[:, :, 1], 90, 0), 1)})")
        print(f"{'  OCA (media de 8 semillas)':30s} {np.round(res[:, :, 0].mean(0)).astype(int)}                {np.round(100 * res[:, :, 1].mean(0), 1)} %"
              f"   (de {np.round(100 * res[:, :, 1].min(0), 1)} a {np.round(100 * res[:, :, 1].max(0), 1)}); {t_oca:.1f} s por selección")
        print(f"  error de la selección (decisiones mal clasificadas, media): {np.round(np.mean([e for _, e in sel], 0), 1)}")
        print(f"  distancia al resultado de las 200 (suma de los tres niveles): OCA media {d_oca.mean():.3f}, peor {d_oca.max():.3f}; al azar mediana {np.median(d_azar):.3f}")
        print(f"  subconjuntos al azar mejores que la OCA media: {(d_azar < d_oca.mean()).mean():.1%}; mejores que la peor OCA: {(d_azar < d_oca.max()).mean():.1%}")
        mejor = int(np.argsort(d_oca)[len(d_oca) // 2])         # la semilla mediana, para la figura
        guardar[K] = dict(azar=azar, res=res, primeras=primeras, S=sel[mejor][0])
        print(f"  geologías elegidas (semilla {mejor}): {sel[mejor][0].tolist()}")

    # el costo: evaluaciones del simulador (decisión × geología)
    n_omo = OMOPSO['nPop'] * (OMOPSO['MaxIter'] + 1)
    print(f"\ncosto en evaluaciones (decisión × geología): OMOPSO robusto con 200 = {n_omo * 200:,}; referencia de la OCA = {len(REF) * 200:,}; "
          f"OMOPSO con 20 = {n_omo * 20:,} y con 10 = {n_omo * 10:,} por cada optimización posterior")

    # ---- figura
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    c200 = curva(np.arange(N)); K = 20
    ax[0].plot(100 * c200[:, 1], 100 * c200[:, 2], "-", color="black", lw=2.2, label="200 geologías")
    for j in range(40):
        c = curva(np.random.default_rng(100 + j).choice(N, K, replace=False))
        ax[0].plot(100 * c[:, 1], 100 * c[:, 2], "-", color="#9AA7B4", lw=0.7, alpha=0.6, label="20 al azar (40 subconjuntos)" if j == 0 else None)
    c = curva(guardar[K]["S"]); ax[0].plot(100 * c[:, 1], 100 * c[:, 2], "o-", color="#C08A0B", lw=2.2, ms=4, label="20 elegidas con la OCA")
    c = curva(np.arange(K)); ax[0].plot(100 * c[:, 1], 100 * c[:, 2], "--", color="#B03A2E", lw=1.5, label="las primeras 20")
    ax[0].plot([0, 40], [0, 40], ":", color="#B03A2E", lw=1.2, label="promesa cumplida"); ax[0].set_xlim(-1, 41); ax[0].set_ylim(0, 70); ax[0].set_xlabel("riesgo PROMETIDO por el conjunto con que se optimizó [%]")
    ax[0].set_ylabel("riesgo REAL en 200 geologías nuevas [%]"); ax[0].set_title("¿Cumple el menú lo que promete?"); ax[0].legend(frameon=False); ax[0].grid(alpha=0.25)
    x = np.arange(3); w = 0.19
    for j, (nombre, v, col) in enumerate((("200 geologías", 100 * completo[:, 1], "black"), ("OCA · 20", 100 * guardar[20]["res"][:, :, 1].mean(0), "#C08A0B"),
                                         ("OCA · 10", 100 * guardar[10]["res"][:, :, 1].mean(0), "#E3C36A"),
                                         ("azar · 20", 100 * np.median(guardar[20]["azar"][:, :, 1], 0), "#7C8B99"), ("azar · 10", 100 * np.median(guardar[10]["azar"][:, :, 1], 0), "#C3CCD5"))):
        b = ax[1].bar(x + (j - 2) * w, v, w, color=col, label=nombre)
        for xi, vi in zip(x + (j - 2) * w, v):
            ax[1].text(xi, vi + 0.7, f"{vi:.0f}", ha="center", fontsize=8)
    ax[1].plot([-0.5, 2.5], [0, 0], color="none")
    for xi, r in zip(x, NIVELES):
        ax[1].plot([xi - 0.5, xi + 0.5], [100 * r] * 2, ":", color="#B03A2E", lw=1.2)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"se pide ≤ {100 * r:.0f} %" for r in NIVELES]); ax[1].set_ylabel("riesgo REAL [%]")
    ax[1].set_title("Lo que se pide (línea punteada) y lo que se obtiene"); ax[1].legend(frameon=False, ncol=2, fontsize=9); ax[1].grid(axis="y", alpha=0.25)
    fig.tight_layout(); salida = os.path.join(S4, "Información", "figuras", "oca_fritz.png"); fig.savefig(salida, dpi=130)
    print(f"\nfigura: {salida}   ({time.time() - t0:.0f} s en total)")


if __name__ == "__main__":
    main()
