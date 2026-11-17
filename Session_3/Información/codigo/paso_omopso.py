"""Dibujos de la sección 2 del cuaderno alterno de la sesión 3: OMOPSO paso a paso.

Solo dibujan lo que el cuaderno ya calculó con las funciones de pyOMOPSO; no cambian el algoritmo.
Trabajan en el espacio de objetivos de un problema de dos objetivos. Por omisión dibujan el frente
verdadero de ZDT1 (con el +0.01 del cuaderno); otro frente se pasa en `verdadero`.
Cada figura lleva arriba la tira de los siete pasos, con el paso que ilustra resaltado.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

AZUL, OCRE, VERDE, GRIS, ROJO, MORADO, TINTA = "#003D79", "#C08A0B", "#1F7A55", "#B8C2C8", "#C0392B", "#6A3D9A", "#1B2B33"
PASOS = ["1 · crear", "2 · no dominadas", "3 · líder", "4 · mover", "5 · mutar", "6 · repositorio", "7 · archivo ε"]
_f1 = np.linspace(0, 1, 200)
ZDT1 = np.column_stack([_f1 + 0.01, 1 - np.sqrt(_f1) + 0.01])


def objetivos(lista):
    """Los objetivos de una lista de partículas como tabla: una fila por partícula, una columna por objetivo."""
    return np.array([np.ravel(p['cost']) for p in lista]) if len(lista) else np.zeros((0, 2))


def _curva(ax, verdadero):
    v = ZDT1 if verdadero is None else verdadero
    ax.plot(v[:, 0], v[:, 1], "-", color=VERDE, lw=2, alpha=0.6, label="frente verdadero")


def _ejes(ax, titulo, xl="$f_1$", yl="$f_2$"):
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(titulo, fontsize=11); ax.legend(fontsize=8.5, loc="upper right")


def _tira(fig, activos):
    """La tira de los siete pasos en la parte alta de la figura; `activos` son los pasos que la figura ilustra."""
    fig.tight_layout(); fig.subplots_adjust(top=0.80)
    n = len(PASOS); ancho = 0.96 / n; letra = 8.5 if fig.get_figwidth() >= 10 else 6.8   # en las figuras angostas, letra más chica
    for k, nombre in enumerate(PASOS):
        on = (k + 1) in activos
        fig.text(0.02 + ancho * (k + 0.5), 0.94, nombre, ha="center", va="center", fontsize=letra, color="white" if on else "#7C8A91",
                 fontweight="bold" if on else "normal",
                 bbox=dict(boxstyle="round,pad=0.35", fc=AZUL if on else "#EEF2F4", ec=AZUL if on else "#D5DDE2"))


def mapa():
    """El mapa de OMOPSO: los pasos 1 y 2 preparan; los pasos 3 a 7 se repiten en cada iteración."""
    fig, ax = plt.subplots(figsize=(12, 3.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")
    cajas = [(0.2, "1 · Crear", "el enjambre al azar\ny evaluarlo", OCRE), (1.9, "2 · No dominadas", "forman el\nrepositorio", OCRE),
             (3.9, "3 · Líder", "el del repositorio\ncon más espacio", AZUL), (5.5, "4 · Mover", "ecuaciones de PSO\nhacia el líder", AZUL),
             (7.1, "5 · Mutar", "y quedarse solo\ncon lo que domina", AZUL), (8.7, "6 · Repositorio", "meter, depurar\ny recortar", AZUL),
             (10.3, "7 · Archivo ε", "la respuesta:\nuna por caja", ROJO)]
    for x, t, s, c in cajas:
        ax.add_patch(FancyBboxPatch((x, 1.1), 1.4, 1.3, boxstyle="round,pad=0.04", fc="white", ec=c, lw=2))
        ax.text(x + 0.7, 2.05, t, ha="center", va="center", fontsize=10, fontweight="bold", color=c)
        ax.text(x + 0.7, 1.52, s, ha="center", va="center", fontsize=8.3, color=TINTA)
    for a, b in zip(cajas[:-1], cajas[1:]):
        ax.add_patch(FancyArrowPatch((a[0] + 1.45, 1.75), (b[0] - 0.05, 1.75), arrowstyle="-|>", mutation_scale=13, color=TINTA, lw=1.2))
    ax.add_patch(FancyArrowPatch((11.0, 1.05), (4.6, 1.05), connectionstyle="arc3,rad=-0.22", arrowstyle="-|>", mutation_scale=15, color=AZUL, lw=1.8))
    ax.text(7.8, 0.12, "se repite en cada iteración (MaxIter veces)", ha="center", fontsize=9.5, color=AZUL, style="italic")
    ax.text(1.75, 2.85, "una sola vez", ha="center", fontsize=9.5, color=OCRE, style="italic")
    ax.text(7.8, 2.85, "el ciclo", ha="center", fontsize=9.5, color=AZUL, style="italic")
    ax.set_title("OMOPSO en siete pasos", fontsize=12); plt.tight_layout(); plt.show()


def enjambre(pop, repo, verdadero=None):
    """Pasos 1 y 2: el enjambre recién creado; dominadas en gris, no dominadas (el repositorio) en ocre."""
    F = objetivos(pop); dom = np.array([p['is_dominated'] for p in pop]) == 1; R = objetivos(repo)
    fig, ax = plt.subplots(figsize=(7.4, 5))
    _curva(ax, verdadero)
    ax.plot(F[dom, 0], F[dom, 1], "o", mfc=GRIS, mec="k", ms=7, label=f"dominadas ({dom.sum()})")
    o = np.argsort(R[:, 0]); ax.plot(R[o, 0], R[o, 1], "--", color=OCRE, lw=1)
    ax.plot(R[:, 0], R[:, 1], "o", mfc=OCRE, mec="k", ms=10, label=f"no dominadas: el repositorio ({len(R)})")
    _ejes(ax, "El enjambre inicial en el espacio de objetivos")
    _tira(fig, (1, 2)); plt.show()


def lider(repo, leader, verdadero=None):
    """Paso 3: el repositorio con la distancia de hacinamiento de cada miembro y el líder elegido."""
    R = objetivos(repo); cd = np.array([p['crowding_distance'] for p in repo], dtype=float); o = np.argsort(R[:, 0])
    L = np.ravel(leader['cost']); es_lider = np.array([np.allclose(r, L) for r in R])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.25, 1]})
    _curva(a1, verdadero)
    a1.plot(R[o, 0], R[o, 1], "o--", color=OCRE, mec="k", ms=9, lw=1, label="repositorio")
    a1.plot(L[0], L[1], "*", color=ROJO, mec="k", ms=22, label="líder de esta iteración")
    for i in o:
        a1.annotate("∞" if np.isinf(cd[i]) else f"{cd[i]:.2f}", R[i], textcoords="offset points", xytext=(7, 5), fontsize=8.5)
    _ejes(a1, "El repositorio y su hacinamiento (el número de cada punto)")
    a2.bar(range(len(o)), [0 if np.isinf(cd[i]) else cd[i] for i in o], color=[ROJO if es_lider[i] else OCRE for i in o], edgecolor="k")
    for k, i in enumerate(o):
        if np.isinf(cd[i]): a2.text(k, 0.02, "∞", ha="center", va="bottom", fontsize=11)
    a2.set_xticks(range(len(o))); a2.set_xticklabels([f"{k + 1}" for k in range(len(o))], fontsize=8)
    a2.set_xlabel("miembro del repositorio (ordenados por $f_1$)"); a2.set_ylabel("distancia de hacinamiento")
    a2.set_title("Se elige el de mayor distancia finita", fontsize=11)
    _tira(fig, (3,)); plt.show()


def movimiento(F_antes, pop, leader, verdadero=None):
    """Paso 4: cada partícula, de donde estaba a donde quedó, en el espacio de objetivos."""
    F = objetivos(pop); L = np.ravel(leader['cost'])
    fig, ax = plt.subplots(figsize=(7.4, 5))
    _curva(ax, verdadero)
    for a, b in zip(F_antes, F):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=AZUL, lw=1, alpha=0.7))
    ax.plot(F_antes[:, 0], F_antes[:, 1], "o", mfc=GRIS, mec="k", ms=6, label="antes")
    ax.plot(F[:, 0], F[:, 1], "o", mfc=AZUL, mec="k", ms=6, label="después")
    ax.plot(L[0], L[1], "*", color=ROJO, mec="k", ms=20, label="líder")
    _ejes(ax, "Todas se mueven con las ecuaciones de PSO, hacia el mismo líder")
    _tira(fig, (4,)); plt.show()


def mutacion(tipos, F_antes, pop, pm, verdadero=None):
    """Paso 5: qué tipo de mutación tocó a cada partícula y cuáles se quedaron con el mutante (las que cambiaron de objetivos)."""
    F = objetivos(pop); tipo = np.ravel(tipos).astype(int); aceptada = np.any(F != F_antes, axis=1)
    nombres = {1: f"radio que se apaga\n(pm = {pm:.2f})", 2: "en todo el rango", 3: "sin mutación"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1, 1.25]})
    cuenta = [int(np.sum(tipo == t)) for t in (1, 2, 3)]; acept = [int(np.sum((tipo == t) & aceptada)) for t in (1, 2, 3)]
    a1.bar([0, 1, 2], cuenta, color=[MORADO, OCRE, GRIS], edgecolor="k", label="les tocó")
    a1.bar([0, 1, 2], acept, color="none", edgecolor="k", hatch="///", label="el mutante dominó y se quedó")
    a1.set_xticks([0, 1, 2]); a1.set_xticklabels([nombres[t] for t in (1, 2, 3)], fontsize=8.5)
    a1.set_ylabel("partículas"); a1.set_ylim(0, max(cuenta) + 2); a1.set_title("Un tercio de cada tipo", fontsize=11); a1.legend(fontsize=8.5, loc="upper right")
    _curva(a2, verdadero)
    a2.plot(F[~aceptada, 0], F[~aceptada, 1], "o", mfc=AZUL, mec="k", ms=6, label="sin cambio")
    for a, b in zip(F_antes[aceptada], F[aceptada]):
        a2.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=MORADO, lw=1.6))
    a2.plot(F[aceptada, 0], F[aceptada, 1], "o", mfc=MORADO, mec="k", ms=8, label=f"se quedó con el mutante ({aceptada.sum()})")
    _ejes(a2, "Solo cambian las que mejoraron")
    _tira(fig, (5,)); plt.show()


def repositorio(repo_antes, repo, archivo, verdadero=None):
    """Pasos 6 y 7: el repositorio antes y después de actualizarlo, y el archivo ε que sale de él."""
    R0, R, A = objetivos(repo_antes), objetivos(repo), objetivos(archivo)
    fig, ax = plt.subplots(figsize=(7.4, 5))
    _curva(ax, verdadero)
    ax.plot(R0[:, 0], R0[:, 1], "o", mfc="none", mec=OCRE, ms=11, label=f"repositorio antes ({len(R0)})")
    ax.plot(R[:, 0], R[:, 1], "o", mfc=OCRE, mec="k", ms=7, label=f"repositorio después ({len(R)})")
    ax.plot(A[:, 0], A[:, 1], "s", mfc="none", mec=ROJO, ms=13, mew=1.8, label=f"archivo ε ({len(A)})")
    _ejes(ax, "El repositorio se depura; el archivo ε guarda la respuesta")
    _tira(fig, (6, 7)); plt.show()


def evolucion(historia, verdadero=None, cortes=(1, 10, 20, -1)):
    """El archivo ε en algunas iteraciones del ciclo completo."""
    fig, axs = plt.subplots(1, len(cortes), figsize=(3.3 * len(cortes), 3.5), sharex=True, sharey=True)
    for ax, c in zip(axs, cortes):
        it = len(historia) if c == -1 else c; A = historia[it - 1]
        _curva(ax, verdadero)
        ax.plot(A[:, 0], A[:, 1], "s", mfc="none", mec=ROJO, ms=6)
        ax.set_title(f"iteración {it}: {len(A)} en el archivo", fontsize=10); ax.set_xlabel("$f_1$")
    axs[0].set_ylabel("$f_2$")
    plt.suptitle("El archivo ε se acerca a la curva verdadera y se reparte por ella", fontsize=11)
    plt.tight_layout(); plt.show()


def comparar(archivo, archivo_caja, verdadero_f2=lambda f1: 1 - np.sqrt(f1 - 0.01) + 0.01):
    """Compara el archivo del ciclo escrito a la vista con el de pyOMOPSO (misma semilla) e imprime el resultado."""
    A, B = objetivos(archivo), objetivos(archivo_caja)
    iguales = A.shape == B.shape and np.allclose(A, B)
    print(f"ciclo a la vista: {len(A)} soluciones  ·  pyOMOPSO: {len(B)} soluciones  ·  ¿el mismo archivo? {'sí' if iguales else 'NO'}")
    print(f"desviación media a la curva verdadera: {np.mean(np.abs(A[:, 1] - verdadero_f2(A[:, 0]))):.4f}")


def particula(pop, i=0):
    """Imprime los datos de la partícula i del enjambre, uno por renglón, con una explicación corta de cada uno."""
    que = {'position': "dónde está: sus variables", 'velocity': "hacia dónde va (nace quieta)", 'cost': "cuánto vale: sus objetivos",
           'best_position': "su mejor posición hasta ahora", 'best_cost': "los objetivos de esa mejor posición",
           'is_dominated': "1 si otra la domina, 0 si no", 'crowding_distance': "el espacio que tiene alrededor"}
    print(f"el enjambre tiene {len(pop)} partículas; la partícula {i} es:")
    for llave, valor in pop[i].items():
        if valor is not None:
            print(f"  {llave:<15} {str(np.round(np.ravel(valor), 3)):<38} {que.get(llave, '')}")
