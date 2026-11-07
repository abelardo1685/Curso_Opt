# -*- coding: utf-8 -*-
"""hidra.py — El acuífero del curso, listo para usarse sin ver el simulador por dentro.

Curso «Optimización metaheurística orientada a modelación hidrogeológica»
UNAM, Instituto de Geología, Laboratorio HIDRA.

El estudiante solo necesita cuatro cosas:

    import hidra
    hidra.mapa()                          # el acuífero, los pozos y el humedal
    r = hidra.simular([300, 300, 300])    # una corrida del simulador con esos bombeos [m³/d]
    r.abatimiento_pozos                   # abatimiento en los 3 pozos [m]
    hidra.mapa(r)                         # el cono de abatimiento de esa corrida

Por debajo trabaja rrwpt (Rodríguez-Pretelín y Nowak, 2018, Advances in Water
Resources): flujo en un acuífero confinado resuelto por elementos finitos en una
malla de 90 por 90 celdas de 15 m. Cada llamada a simular() resuelve el sistema
completo con los tres pozos a la vez; no usa ninguna tabla precalculada.
"""
import os
import sys


def _paquetes_del_venv(raiz):
    """Carpetas site-packages del .venv del curso (junto a este archivo), si su versión de Python es la que corre ahora."""
    cfg = os.path.join(raiz, ".venv", "pyvenv.cfg")
    if not os.path.isfile(cfg):
        return []
    version = ""
    with open(cfg, encoding="utf-8", errors="ignore") as f:
        for linea in f:
            clave, _, valor = linea.partition("=")
            if clave.strip() in ("version", "version_info"):
                version = valor.strip()
    mayor_menor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if ".".join(version.split(".")[:2]) != mayor_menor:
        return []
    candidatas = [os.path.join(raiz, ".venv", "Lib", "site-packages"),
                  os.path.join(raiz, ".venv", "lib", f"python{mayor_menor}", "site-packages")]
    return [c for c in candidatas if os.path.isdir(c)]


# --- localizar rrwpt: el paquete compilado que instala requirements.txt (carpeta wheels/) en el .venv del curso.
#     Si el cuaderno corre con otro Python (VS Code eligió un kernel que no es .venv), se usan los paquetes del .venv
#     que está junto a este archivo, antes de importar numpy, para que todo salga del mismo entorno.
try:
    import rrwpt  # noqa: F401
except ImportError:  # pragma: no cover
    import site
    _aqui = os.path.dirname(os.path.abspath(__file__))
    _venv = _paquetes_del_venv(_aqui)
    if _venv:
        _antes = list(sys.path)
        for _sp in _venv:
            site.addsitedir(_sp)                       # también lee los .pth del entorno
        sys.path[:] = [p for p in sys.path if p not in _antes] + _antes
        print(f"Aviso: este cuaderno corre con {sys.executable}, no con el .venv del curso.\n"
              "hidra usará los paquetes del .venv. Para quitar este aviso, elige el kernel .venv (arriba a la derecha).")
    try:
        import rrwpt  # noqa: F401
    except ImportError:
        raise ImportError(
            f"No se encontró rrwpt. Este cuaderno corre con {sys.executable}.\n"
            "Elige el kernel .venv del curso (arriba a la derecha del cuaderno, «Select Kernel» → «Python Environments» → .venv)\n"
            "o vuelve a instalar la paquetería con la guía de instalación (sección 7).") from None

import numpy as np

from rrwpt import config as _config, fem as _fem, geostat as _geostat, pipeline as _pipeline

# ============================================================== el acuífero del curso
LADO = 90 * 15.0                     # [m] lado del dominio cuadrado
N_CELDAS = 90                        # celdas por lado
SEG_POR_DIA = 86400.0                # el modelo trabaja en m³/s; el curso en m³/d

NOMBRES_POZOS = ["Pozo 1", "Pozo 2", "Pozo 3"]
POZOS_FRACCION = [(0.35, 0.55), (0.55, 0.35), (0.60, 0.65)]   # (x, y) como fracción del lado
HUMEDAL_FRACCION = (0.45, 0.45)                              # el humedal protegido (x, y)

S_MAX = 3.0                   # [m] abatimiento máximo permitido en cada pozo
S_MAX_HUMEDAL = 1.25          # [m] abatimiento máximo permitido en el humedal (protección del ecosistema)
P3_COMPROMETIDO = 300.0       # [m³/d] demanda ya comprometida del pozo 3
P_MAX = 1000.0                # [m³/d] cota superior de cualquier bombeo (capacidad de la bomba)

SEMILLAS_GEOLOGIA = list(range(100, 110))   # las 10 geologías del ensamble (Sesión 4)

_CONFIG = {
    'Two_D': 1, 'n_pts_x': N_CELDAS, 'n_pts_y': N_CELDAS, 'd_pts_x': 15.0, 'd_pts_y': 15.0,
    'het': 1, 'tim.tend': 60, 'tim.deltQS': 10, 'crit.t_crit': 30,
    'trns.npartic_QS': 500,
    'inP': list(POZOS_FRACCION[0]), 'Ex_pump': 1,
    'stru.variA': 0.5, 'stru.variE': 0.5, 'use_var_matern_kappa': 0,
}

_ESTADO = {}


def _preparar():
    """Construye malla, modelo geoestadístico y fronteras una sola vez."""
    if _ESTADO:
        return _ESTADO
    ctrl = _config.make_ctrl(**_CONFIG, **{'Ex_inP': np.array(POZOS_FRACCION[1:], dtype=float)})
    grid, flowpar, model, ctrl = _pipeline.setup(ctrl, np.random.default_rng(42))
    campos = [_geostat.uncsim_kfield(ctrl, model, grid, rng=np.random.default_rng(s))
              for s in SEMILLAS_GEOLOGIA]
    yk_hom = np.full(grid.data.nel, float(np.mean([c.mean() for c in campos])))
    _ESTADO.update(ctrl=ctrl, grid=grid, flowpar=flowpar, model=model,
                   campos=campos, yk_hom=yk_hom,
                   nodos_pozos=np.ravel(ctrl.bc.well_pts)[:3].astype(int),
                   n_pts=tuple(int(n) for n in grid.data.n_pts))
    return _ESTADO


def _nodo(fraccion_xy):
    """Índice de nodo de un punto dado como fracción (x, y), con la misma regla que usa rrwpt para los pozos."""
    e = _preparar()
    n_el = e["grid"].data.n_el
    iy = int(np.round(n_el[0] * fraccion_xy[1])) - 1
    ix = int(np.round(n_el[1] * fraccion_xy[0])) - 1
    return int(np.ravel_multi_index((iy, ix), e["n_pts"], order="F"))


def _xy_de_nodo(nodo):
    e = _preparar()
    iy, ix = np.unravel_index(int(nodo), e["n_pts"], order="F")
    d = e["grid"].data.d_pts
    return float(ix * d[1]), float(iy * d[0])


def pozos_xy():
    """Coordenadas (x, y) en metros de los tres pozos."""
    e = _preparar()
    return [_xy_de_nodo(n) for n in e["nodos_pozos"]]


def humedal_xy():
    """Coordenadas (x, y) en metros del humedal."""
    return _xy_de_nodo(_nodo(HUMEDAL_FRACCION))


# ============================================================== el simulador
class Resultado:
    """Lo que devuelve una corrida del simulador."""

    def __init__(self, bombeos, campo_vector):
        e = _preparar()
        self.bombeos = np.asarray(bombeos, dtype=float)
        self._vector = campo_vector                                   # abatimiento en los 8 281 nodos [m]
        self.campo = campo_vector.reshape(e["n_pts"], order="F")      # matriz [fila = y, columna = x]
        self.abatimiento_pozos = campo_vector[e["nodos_pozos"]]
        self.abatimiento_humedal = float(campo_vector[_nodo(HUMEDAL_FRACCION)])
        self.abatimiento_maximo = float(campo_vector.max())

    def en(self, x, y):
        """Abatimiento [m] en el nodo más cercano al punto (x, y) en metros."""
        e = _preparar()
        d = e["grid"].data.d_pts
        iy = int(np.clip(round(y / d[0]), 0, e["n_pts"][0] - 1))
        ix = int(np.clip(round(x / d[1]), 0, e["n_pts"][1] - 1))
        return float(self.campo[iy, ix])

    def __repr__(self):
        p = ", ".join(f"{b:.1f}" for b in self.bombeos)
        s = ", ".join(f"{v:.2f}" for v in self.abatimiento_pozos)
        return (f"Resultado(bombeos=[{p}] m³/d → abatimientos en pozos=[{s}] m, "
                f"humedal={self.abatimiento_humedal:.2f} m)")


def simular(bombeos, geologia=None):
    """Una corrida del simulador. bombeos = [p1, p2, p3] en m³/d.

    Resuelve la ecuación de flujo del acuífero confinado con los tres pozos
    bombeando a la vez (elementos finitos, 8 281 nodos). Devuelve un Resultado.
    geologia = None usa el acuífero del curso (homogéneo equivalente);
    geologia = 0..9 usa una de las diez geologías heterogéneas del ensamble.
    """
    e = _preparar()
    bombeos = np.asarray(bombeos, dtype=float).reshape(-1)
    if bombeos.size != 3:
        raise ValueError("simular() espera exactamente tres bombeos: [p1, p2, p3] en m³/d")
    if np.any(bombeos < 0):
        raise ValueError("los bombeos no pueden ser negativos (este curso no inyecta agua)")
    yk = e["yk_hom"] if geologia is None else e["campos"][int(geologia)]
    K = np.exp(np.asarray(yk, dtype=float).reshape(-1))
    grid, ctrl = e["grid"], e["ctrl"]
    M_h_d, resolver = _fem._get_system(K, ctrl.bc.fix_pts_h, grid)
    fuente = np.zeros(grid.data.npts)
    for nodo, q in zip(e["nodos_pozos"], bombeos / SEG_POR_DIA):   # m³/d → m³/s
        fuente[nodo] -= q
    fuente[ctrl.bc.fix_pts_h] = 0.0            # carga fija (sin abatimiento) en las fronteras izquierda y derecha
    h = resolver(fuente)                        # cambio de carga: negativo donde se bombea
    return Resultado(bombeos, -h)               # abatimiento = -cambio de carga


def matriz_respuesta(geologia=None):
    """La matriz 3×3 de coeficientes de respuesta α [m por m³/d] en los pozos, fabricada con tres corridas unitarias."""
    columnas = []
    for j in range(3):
        unit = np.zeros(3); unit[j] = 1.0
        columnas.append(simular(unit, geologia).abatimiento_pozos)
    return np.column_stack(columnas)


def coeficientes_en(x, y, geologia=None):
    """Fila de coeficientes de respuesta [m por m³/d] en el punto (x, y) en metros: cuánto abate ahí cada pozo por unidad de bombeo."""
    fila = []
    for j in range(3):
        unit = np.zeros(3); unit[j] = 1.0
        fila.append(simular(unit, geologia).en(x, y))
    return np.array(fila)


def coeficientes_humedal(geologia=None):
    return coeficientes_en(*humedal_xy(), geologia)


def campo_respuesta(geologia=None):
    """Respuesta unitaria en TODO el acuífero: arreglo (91, 91, 3), un mapa por pozo."""
    return np.stack([simular(np.eye(3)[j], geologia).campo for j in range(3)], axis=-1)


SEMILLA_BASE_ENSAMBLE = {"optimizacion": 1000, "validacion": 5000}   # Sesión 4: dos ensambles grandes e independientes


def _semilla_ensamble(tipo, k):
    """La semilla de la geología k del ensamble: cada geología tiene la suya, así que las primeras n son siempre las mismas
    aunque se pidan más. Optimización: 1000 + k (k < 4000) y 1 000 000 + k después, para no encimarse con validación (5000 + k)."""
    if tipo == "optimizacion":
        return SEMILLA_BASE_ENSAMBLE[tipo] + k if k < 4000 else 1_000_000 + k
    return SEMILLA_BASE_ENSAMBLE[tipo] + k


def _generar_ensamble(tipo, k0, k1, verbose=True):
    """Las geologías k0, …, k1 − 1 del ensamble `tipo`, con el simulador (una factorización por geología)."""
    import time
    e = _preparar()
    grid, ctrl = e["grid"], e["ctrl"]
    nodo_h = _nodo(HUMEDAL_FRACCION)
    n_el = tuple(int(v) for v in grid.data.n_el)
    n = k1 - k0
    alpha = np.zeros((n, 3, 3)); alpha_h = np.zeros((n, 3)); logK = np.zeros((n,) + n_el, dtype=np.float32)
    n_pts = tuple(int(v) for v in grid.data.n_pts)
    idx = np.arange(0, n_pts[0], 6)                                  # 16 x 16 nodos, cada 90 m
    IY, IX = np.meshgrid(idx, idx, indexing="ij")
    nodos_malla = np.ravel_multi_index((IY.ravel(), IX.ravel()), n_pts, order="F")
    d = grid.data.d_pts
    xy_malla = np.column_stack([IX.ravel() * d[1], IY.ravel() * d[0]])
    alpha_malla = np.zeros((n, nodos_malla.size, 3), dtype=np.float32)
    t0 = time.time()
    for i, k in enumerate(range(k0, k1)):
        yk = _geostat.uncsim_kfield(ctrl, e["model"], grid, rng=np.random.default_rng(_semilla_ensamble(tipo, k)))
        _, resolver = _fem._get_system(np.exp(np.asarray(yk, dtype=float).reshape(-1)), ctrl.bc.fix_pts_h, grid)
        for j, nodo in enumerate(e["nodos_pozos"]):
            fuente = np.zeros(grid.data.npts); fuente[nodo] = -1.0 / SEG_POR_DIA
            fuente[ctrl.bc.fix_pts_h] = 0.0
            s = -resolver(fuente)
            alpha[i, :, j] = s[e["nodos_pozos"]]; alpha_h[i, j] = s[nodo_h]; alpha_malla[i, :, j] = s[nodos_malla]
        logK[i] = np.asarray(yk, dtype=float).reshape(n_el, order="F")
        if verbose and (i + 1) % 50 == 0:
            falta = (time.time() - t0) / (i + 1) * (n - i - 1)
            print(f"  ensamble de {tipo}: geología {k + 1} de {k1} ({time.time() - t0:.0f} s; faltan unos {falta:.0f} s)", flush=True)
    return {"alpha": alpha, "alpha_h": alpha_h, "logK": logK, "alpha_malla": alpha_malla, "xy_malla": xy_malla}


def ensamble(n=200, tipo="optimizacion", verbose=True):
    """Un ensamble grande de geologías (Sesión 4): n campos de ln K con el mismo modelo geoestadístico de las diez del curso.

    tipo = "optimizacion" o "validacion": dos ensambles independientes, uno para optimizar y otro para comprobar la decisión en
    geologías que el optimizador nunca vio. Cada geología k sale de su propia semilla (ver _semilla_ensamble): las primeras n
    geologías son siempre las mismas, en cualquier computadora, aunque después se pidan más. Así, en clase y en la tarea se
    usan 200, y en casa se puede subir n (500, 1000, 5000…) para ver qué cambia.
    Para cada geología se resuelve el flujo con los tres bombeos unitarios (una factorización por geología) y se guardan:
      alpha   (n, 3, 3)  matriz de respuesta en los pozos [m por m³/d]
      alpha_h (n, 3)     fila del humedal
      logK    (n, 90, 90) el campo, para dibujarlo
      alpha_malla (n, 256, 3) respuesta unitaria en una malla gruesa de 16 x 16 puntos (para mapas de probabilidad y OCA)
      xy_malla (256, 2)   coordenadas [m] de esa malla
    El resultado se guarda en _cache_hidra/ junto a este archivo. La primera vez tarda unos 0.13 s por geología; si ya existe
    un ensamble más grande se recorta, y si existe uno más chico solo se generan las geologías que faltan.
    """
    import glob
    import re
    if tipo not in SEMILLA_BASE_ENSAMBLE:
        raise ValueError("tipo debe ser 'optimizacion' o 'validacion'")
    n = int(n)
    if n < 1:
        raise ValueError("n debe ser al menos 1")
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache_hidra")
    ruta = os.path.join(carpeta, f"ensamble_{tipo}_{n}.npz")
    if os.path.exists(ruta):
        d = np.load(ruta)
        return {k: d[k] for k in d.files}
    guardados = sorted((int(re.search(r"_(\d+)\.npz$", f).group(1)), f) for f in glob.glob(os.path.join(carpeta, f"ensamble_{tipo}_*.npz")))
    mayor = next(((m, f) for m, f in guardados if m >= n), None)
    if mayor is not None:                                            # ya hay uno más grande: las primeras n son las mismas
        d = np.load(mayor[1])
        return {k: (d[k] if k == "xy_malla" else d[k][:n]) for k in d.files}
    base = {}
    if guardados:                                                    # hay uno más chico: solo se generan las que faltan
        m, f = guardados[-1]
        d = np.load(f); base = {k: d[k] for k in d.files}
    k0 = len(base["alpha"]) if base else 0
    if verbose:
        print(f"Generando con el simulador las geologías {k0 + 1} a {n} del ensamble de {tipo} (unos {0.13 * (n - k0):.0f} s)…", flush=True)
    nuevo = _generar_ensamble(tipo, k0, n, verbose=verbose)
    res = {k: (nuevo[k] if k == "xy_malla" else (np.concatenate([base[k], nuevo[k]]) if base else nuevo[k])) for k in nuevo}
    os.makedirs(carpeta, exist_ok=True)
    np.savez_compressed(ruta, **res)
    return res


def campo_logK(geologia=None):
    """La geología: el campo de log-conductividad Y = ln K como matriz [fila = y, columna = x].
    geologia = None es el acuífero homogéneo del curso; 0..9, una de las diez geologías del ensamble."""
    e = _preparar()
    yk = e["yk_hom"] if geologia is None else e["campos"][int(geologia)]
    return np.asarray(yk, dtype=float).reshape(tuple(int(n) for n in e["grid"].data.n_el), order="F")


# ============================================================== gráficas
def mapa(resultado=None, titulo=None, ax=None, mostrar_humedal=True, niveles=None, vmax=None):
    """Dibuja el acuífero. Sin argumentos: solo la geometría. Con un Resultado: el cono de abatimiento.
    vmax fija el tope de la escala de color (para comparar varios mapas con la misma escala)."""
    import matplotlib.pyplot as plt
    propio = ax is None
    if propio:
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ext = [0, LADO, 0, LADO]
    if resultado is not None:
        im = ax.imshow(resultado.campo, origin="lower", extent=ext, cmap="Blues", vmin=0,
                       vmax=vmax if vmax is not None else max(0.5, float(np.ceil(resultado.abatimiento_maximo * 2) / 2)))
        cs = ax.contour(np.linspace(0, LADO, resultado.campo.shape[1]), np.linspace(0, LADO, resultado.campo.shape[0]),
                        resultado.campo, levels=niveles or [0.25, 0.5, 1, 2, 3], colors="k", linewidths=0.6)
        ax.clabel(cs, fmt="%g m", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="abatimiento [m]")
    else:
        ax.set_xlim(0, LADO); ax.set_ylim(0, LADO)
        ax.set_facecolor("#EEF4F6")
    # fronteras de carga fija: izquierda (entrada regional) y derecha (arroyo)
    ax.plot([0, 0], [0, LADO], color="#54696F", lw=4, solid_capstyle="butt")
    ax.plot([LADO, LADO], [0, LADO], color="#2E9BB5", lw=5, solid_capstyle="butt")
    ax.text(LADO - 25, LADO / 2, "arroyo (carga fija)", rotation=90, va="center", ha="right", fontsize=8, color="#0B5B72")
    ax.text(25, LADO / 2, "entrada regional (carga fija)", rotation=90, va="center", ha="left", fontsize=8, color="#54696F")
    for (x, y), nombre, b in zip(pozos_xy(), NOMBRES_POZOS,
                                 (resultado.bombeos if resultado is not None else [None] * 3)):
        ax.plot(x, y, "o", ms=9, mfc="white", mec="k", zorder=5)
        etiqueta = nombre if b is None else f"{nombre}\n{b:.0f} m³/d"
        ax.annotate(etiqueta, (x, y), xytext=(8, 8), textcoords="offset points", fontsize=8, fontweight="bold")
    if mostrar_humedal:
        hx, hy = humedal_xy()
        ax.plot(hx, hy, "P", ms=12, mfc="#8fd18f", mec="k", zorder=5)
        ax.annotate("humedal", (hx, hy), xytext=(8, -14), textcoords="offset points", fontsize=8, color="#2d6a2d", fontweight="bold")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    ax.set_title(titulo or ("El acuífero de Fritz Wasser" if resultado is None else
                            f"Abatimiento con bombeos {np.round(resultado.bombeos).astype(int).tolist()} m³/d"))
    if propio:
        plt.tight_layout()
    return ax


def mapa_geologia(geologia=None, titulo=None, ax=None, vmin=None, vmax=None):
    """Dibuja la geología (ln K) con los pozos y el humedal."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
    Y = campo_logK(geologia)
    im = ax.imshow(Y, origin="lower", extent=[0, LADO, 0, LADO], cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="ln K", shrink=0.85)
    for k, (x, y) in enumerate(pozos_xy()):
        ax.plot(x, y, "o", ms=8, mfc="yellow", mec="k", zorder=5)
        ax.annotate(f"pozo {k + 1}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
    hx, hy = humedal_xy()
    ax.plot(hx, hy, "P", ms=10, mfc="#8fd18f", mec="k", zorder=5)
    ax.set_xlim(0, LADO); ax.set_ylim(0, LADO); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(titulo or ("acuífero homogéneo del curso" if geologia is None else f"geología {geologia}"))
    return ax


def graficar_region_factible(A, b, nombres, p_optimo=None, p_max=P_MAX, etiquetas_ejes=("$p_1$ [m³/d]", "$p_2$ [m³/d]"),
                             puntos=None, titulo="Región factible"):
    """Dibuja las restricciones A·[p1, p2] ≤ b en el plano (p1, p2), sombrea la región factible y marca el óptimo."""
    import matplotlib.pyplot as plt
    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float)
    p1 = np.linspace(0, p_max, 400)
    colores = ["#d1633a", "#8a6bbf", "#5a8f5a", "#c08a0b", "#2e9bb5", "#1b2b33"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, ((a1, a2), bk, nombre) in enumerate(zip(A, b, nombres)):
        c = colores[k % len(colores)]
        if abs(a2) > 1e-12:
            ax.plot(p1, (bk - a1 * p1) / a2, color=c, lw=2, label=nombre)
        else:
            ax.axvline(bk / a1, color=c, lw=2, label=nombre)
    P1, P2 = np.meshgrid(p1, np.linspace(0, p_max, 400))
    factible = np.ones_like(P1, dtype=bool)
    for (a1, a2), bk in zip(A, b):
        factible &= (a1 * P1 + a2 * P2 <= bk)
    ax.contourf(P1, P2, factible, levels=[0.5, 1.5], colors=["#9ec9e0"], alpha=0.5)
    if puntos:
        for (x, y), nombre in puntos:
            ax.plot(x, y, "s", ms=7, mfc="white", mec="k", zorder=4)
            ax.annotate(nombre, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
    if p_optimo is not None:
        ax.plot(p_optimo[0], p_optimo[1], "*", ms=22, mfc="#f2c230", mec="k", zorder=5)
        ax.annotate(f"ÓPTIMO ({p_optimo[0]:.1f}, {p_optimo[1]:.1f})", (p_optimo[0], p_optimo[1]),
                    xytext=(12, -18), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.set_xlim(0, p_max); ax.set_ylim(0, p_max)
    ax.set_xlabel(etiquetas_ejes[0]); ax.set_ylabel(etiquetas_ejes[1]); ax.set_title(titulo)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return ax


def frente_3d(*frentes, ejes=("f1", "f2", "f3"), titulo=None, punto=None, normalizar=False, alto=600, camara=(1.5, -1.7, 0.9)):
    """Uno o varios frentes de Pareto en 3-D que se giran con el mouse (plotly).

    Cada frente es una tupla (F, nombre) o (F, nombre, color) o (F, nombre, color, tamaño), con F de forma (n, 3):
    una fila por solución, una columna por objetivo. `punto` = (xyz, nombre) resalta una solución (en rojo).
    Con normalizar=True cada objetivo se lleva a [0, 1] con el mínimo y el máximo del primer frente.

        fig = hidra.frente_3d((F, "Frente de Pareto"), ejes=("Abatimiento", "Costo", "Índice"))
        fig.show()

    Arrastra para girar, rueda del mouse para acercar, doble clic para volver a la vista inicial y pasa el
    cursor por un punto para leer sus tres objetivos. El botón de cámara descarga la vista como PNG.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as e:  # pragma: no cover
        raise ImportError("Falta plotly (la figura 3-D interactiva):  pip install -r requirements.txt") from e
    # Solo el mimetype de plotly: lo dibujan la extensión "Jupyter Notebook Renderers" de VS Code y JupyterLab, también al
    # reabrir el cuaderno. El respaldo HTML de plotly (require.js) es justo lo que deja salidas en blanco en VS Code
    # (microsoft/vscode-jupyter #6404). En Colab se deja el renderer que plotly elige.
    if "google.colab" not in sys.modules:
        pio.renderers.default = "plotly_mimetype"
    paleta = ["#2e9b4f", "black", "#B03A2E", "#1f5fbf", "#c08a0b", "#8a6bbf"]
    F0 = np.asarray(frentes[0][0], dtype=float)
    lo, hi = (F0.min(0), F0.max(0)) if normalizar else (np.zeros(3), np.ones(3))
    fig = go.Figure()
    for k, fr in enumerate(frentes):
        F = (np.asarray(fr[0], dtype=float) - lo) / (hi - lo)
        nombre = fr[1] if len(fr) > 1 else f"frente {k + 1}"
        color = fr[2] if len(fr) > 2 and fr[2] else paleta[k % len(paleta)]
        tam = fr[3] if len(fr) > 3 else 4
        fig.add_trace(go.Scatter3d(x=F[:, 0], y=F[:, 1], z=F[:, 2], mode="markers", name=nombre,
                                   marker=dict(size=tam, color=color, opacity=0.85 if tam > 2 else 0.35,
                                               line=dict(color="black", width=0.5) if tam > 2 else None),
                                   hovertemplate=f"{ejes[0]} %{{x:.3g}}<br>{ejes[1]} %{{y:.3g}}<br>{ejes[2]} %{{z:.3g}}<extra>{nombre}</extra>"))
    if punto is not None:
        xyz = (np.asarray(punto[0], dtype=float) - lo) / (hi - lo)
        fig.add_trace(go.Scatter3d(x=[xyz[0]], y=[xyz[1]], z=[xyz[2]], mode="markers", name=punto[1] if len(punto) > 1 else "elegida",
                                   marker=dict(size=9, color="red", symbol="diamond", line=dict(color="black", width=1))))
    fig.update_layout(title=titulo, height=alto, margin=dict(l=0, r=0, t=40 if titulo else 10, b=0),
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
                      scene=dict(xaxis_title=ejes[0], yaxis_title=ejes[1], zaxis_title=ejes[2], aspectmode="cube",
                                 camera=dict(eye=dict(x=camara[0], y=camara[1], z=camara[2]))))
    return fig



def caras_frente(*frentes, ejes=("f1", "f2", "f3"), titulo=None, punto=None, alto=4.6):
    """Las tres caras de uno o varios frentes de tres objetivos, en 2-D: (f1, f2), (f1, f3) y (f2, f3).

    Es la compañera de frente_3d y se llama igual: cada frente es una tupla (F, nombre) o (F, nombre, color) o
    (F, nombre, color, tamaño), con F de forma (n, 3). `punto` = (xyz, nombre) resalta una solución (rombo rojo).
    Cada cara es lo que se ve al mirar la figura 3-D de frente, desde un lado y desde arriba.

        hidra.caras_frente((F, "Frente de Pareto"), ejes=("Abatimiento", "Costo", "Índice")); plt.show()
    """
    import matplotlib.pyplot as plt
    paleta = ["#2e9b4f", "black", "#B03A2E", "#1f5fbf", "#c08a0b", "#8a6bbf"]
    fig, ejes_mpl = plt.subplots(1, 3, figsize=(17, alto))
    for ax, (i, j) in zip(ejes_mpl, [(0, 1), (0, 2), (1, 2)]):
        for k, fr in enumerate(frentes):
            F = np.asarray(fr[0], dtype=float)
            nombre = fr[1] if len(fr) > 1 else f"frente {k + 1}"
            color = fr[2] if len(fr) > 2 and fr[2] else paleta[k % len(paleta)]
            tam = fr[3] if len(fr) > 3 else 4
            borde = "black" if color in ("white", "#ffffff") else "none"
            ax.scatter(F[:, i], F[:, j], s=3 * tam, color=color, edgecolors=borde, linewidths=0.4, alpha=0.75 if tam > 2 else 0.35, label=nombre)
        if punto is not None:
            xyz = np.asarray(punto[0], dtype=float)
            ax.plot(xyz[i], xyz[j], "D", ms=9, mfc="red", mec="black", label=punto[1] if len(punto) > 1 else "elegida")
        ax.set_xlabel(ejes[i]); ax.set_ylabel(ejes[j]); ax.grid(alpha=0.3)
        ax.set_title(f"{ejes[j]} contra {ejes[i].lower()}")
    ejes_mpl[0].legend(fontsize=8)
    if titulo:
        fig.suptitle(titulo)
    fig.tight_layout()
    return fig

def datos_tarea(identificador):
    """Los datos personales de la tarea, derivados de un identificador del estudiante (siempre los mismos para el mismo texto).

    El identificador es el correo con el que se inscribió al curso (o la clave que le asigne el instructor): cualquier
    texto de al menos 5 caracteres. Se normaliza a minúsculas y sin espacios, para que «Ana@Gmail.com» y «ana@gmail.com»
    den los mismos datos. Devuelve el límite de abatimiento del humedal (1.20 a 1.50 m) y la demanda comprometida del
    pozo 3 (200 a 350 m³/d). Las 49 combinaciones tienen óptimo con los dos pozos libres encendidos.
    Desde la Tarea 2 también devuelve el precio del agua (9.3 a 9.6 $/m³) y una geología del ensamble (1, 3, 4, 7, 8 o 9:
    las que tienen solución con cualquier combinación de datos).
    """
    import hashlib
    clave = "".join(str(identificador).lower().split())
    if len(clave) < 5 or clave.startswith("..."):
        raise ValueError("escribe tu identificador: el correo con el que te inscribiste, entre comillas (por ejemplo 'ana.lopez@gmail.com')")
    h = hashlib.sha256(clave.encode("utf-8")).digest()
    s_hum = round(1.20 + 0.05 * (h[0] % 7), 2)        # 1.20, 1.25, …, 1.50
    p3 = 200.0 + 25.0 * (h[1] % 7)                     # 200, 225, …, 350
    precio = round(9.3 + 0.1 * (h[2] % 4), 2)          # 9.3 … 9.6 $/m³ (Tarea 2: con cualquiera de ellos el óptimo de ganancia es interior)
    geologia = [1, 3, 4, 7, 8, 9][h[3] % 6]             # Tarea 2: una geología del ensamble con solución para todos los datos
    riesgo = [0.05, 0.10, 0.15][h[4] % 3]                # Tarea 4: el riesgo de incumplir que el estudiante acepta
    return {"identificador": clave, "s_max_humedal": s_hum, "p3": p3, "precio_agua": precio, "geologia": geologia,
            "riesgo_aceptable": riesgo}


def verificar_entorno():
    """Imprime versiones y hace una corrida de prueba. Es la primera celda de cada cuaderno."""
    import time
    import scipy, matplotlib
    try:
        import plotly; _plotly = f" | plotly {plotly.__version__}"
    except ImportError:
        _plotly = " | plotly: FALTA (figuras 3-D interactivas): pip install -r requirements.txt"
    try:
        import otter; _plotly += f" | otter {otter.__version__}"
    except ImportError:
        _plotly += " | otter: FALTA (las comprobaciones de las tareas): pip install -r requirements.txt"
    print(f"Python {sys.version.split()[0]} | numpy {np.__version__} | scipy {scipy.__version__} | matplotlib {matplotlib.__version__}{_plotly}")
    t0 = time.time(); r = simular([100, 100, 100]); dt = time.time() - t0
    print(f"Simulador listo: una corrida tarda {dt:.2f} s. {r}")
    print("ENTORNO LISTO")
