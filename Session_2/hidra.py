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


# ============================================================== gráficas del enjambre (sesión 2, versión 3 del cuaderno: 2026-11-05)
# Cada función recibe lo que el algoritmo ya produjo (pop, gbest, pso.historial, la historia de posiciones) y dibuja; no calcula nada del PSO.
def _malla_funcion(f, lim, n=200):
    """Malla (X, Y, Z) de una función de prueba f(x) en el cuadro [lim[0], lim[1]]²."""
    v = np.linspace(lim[0], lim[1], n); X, Y = np.meshgrid(v, v)
    Z = np.array([[f(np.array([x, y])) for x in v] for y in v])
    return X, Y, Z


def superficie_prueba(f, lim=(-5, 5), gbest=None, nombre="Sphere", minimo=(0, 0)):
    """La función de prueba en 3-D y desde arriba, con el mínimo global (estrella) y, si se da, el gbest del enjambre (cruz)."""
    import matplotlib.pyplot as plt
    X, Y, Z = _malla_funcion(f, lim, 150)
    fig = plt.figure(figsize=(13, 5))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot_surface(X, Y, Z, cmap="turbo", edgecolor="none", alpha=0.95)
    ax3.set_title(f"{nombre} en 2-D"); ax3.set_xlabel("$x_1$"); ax3.set_ylabel("$x_2$"); ax3.set_zlabel(f"{nombre}(x)")
    ax = fig.add_subplot(1, 2, 2)
    cf = ax.contourf(X, Y, Z, levels=40, cmap="turbo"); fig.colorbar(cf, ax=ax, label=f"{nombre}(x)")
    ax.plot(minimo[0], minimo[1], "*", ms=18, mfc="yellow", mec="k", label=f"mínimo global {tuple(minimo)}")
    if gbest is not None:
        ax.plot(gbest["position"][0], gbest["position"][1], "wX", ms=12, mec="k", label=f"gbest del enjambre: costo {gbest['cost']:.1e}")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Vista desde arriba"); ax.legend(loc="upper right")
    plt.tight_layout(); plt.show()


def terreno_parametros(f, VarMin, VarMax, parametros, nombre="Sphere", minimo=(0, 0)):
    """El dominio [VarMin, VarMax]² sobre la función de prueba y una ficha con los parámetros del enjambre (un diccionario nombre -> valor)."""
    import matplotlib.pyplot as plt
    X, Y, Z = _malla_funcion(f, (VarMin - 1, VarMax + 1), 200)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.contourf(X, Y, Z, levels=30, cmap="turbo", alpha=0.85)
    ax.add_patch(plt.Rectangle((VarMin, VarMin), VarMax - VarMin, VarMax - VarMin, fill=False, ec="w", lw=2.5, ls="--"))
    ax.plot(minimo[0], minimo[1], "*", ms=18, mfc="yellow", mec="k"); ax.text(minimo[0] + 0.25, minimo[1] + 0.25, f"mínimo de {nombre}: {tuple(minimo)}", color="w", fontsize=10)
    ax.text(VarMin, VarMax + 0.2, f"dominio [{VarMin}, {VarMax}]²: VarMin y VarMax", color="w", fontsize=10)
    ficha = "\n".join(f"{k} = {v}" for k, v in parametros.items())
    ax.text(1.02, 0.97, ficha, transform=ax.transAxes, va="top", fontsize=11, family="monospace", bbox=dict(boxstyle="round", fc="#F2F5F9", ec="#B8D2DA"))
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("El terreno y los parámetros"); plt.tight_layout(); plt.show()


def poblacion(pop, gbest, f, VarMin, VarMax, n_tabla=7, nombre="Sphere"):
    """La «librería» de partículas: a la izquierda, las partículas numeradas sobre el terreno (gbest como estrella); a la derecha, la tabla de pop
    (las primeras n_tabla y la que es gbest, marcada con ★)."""
    import matplotlib.pyplot as plt
    X, Y, Z = _malla_funcion(f, (VarMin, VarMax), 200)
    P0 = np.array([p["position"] for p in pop]); n = len(pop)
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
    ax.contourf(X, Y, Z, levels=30, cmap="turbo", alpha=0.85)
    ax.plot(P0[:, 0], P0[:, 1], "wo", ms=8, mec="k", label="partículas")
    for i, p in enumerate(pop):
        ax.annotate(str(i), p["position"], xytext=(5, 4), textcoords="offset points", fontsize=8, color="w")
    ax.plot(gbest["position"][0], gbest["position"][1], "*", ms=20, mfc="yellow", mec="k", label=f"gbest: costo {gbest['cost']:.3f}")
    ax.set_xlim(VarMin, VarMax); ax.set_ylim(VarMin, VarMax); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"{n} partículas sobre {nombre}"); ax.legend(loc="upper right", fontsize=9)
    axt.axis("off")
    kg = int(np.argmin([p["best_cost"] for p in pop]))
    vec = lambda v: "(" + ", ".join(f"{x:.2f}" for x in v) + ")"
    fila = lambda i, p: [i, vec(p["position"]), vec(p["velocity"]), f"{p['cost']:.3f}", vec(p["best_position"]), f"{p['best_cost']:.3f}"]
    filas = [fila(i, p) for i, p in enumerate(pop[:n_tabla])] + [["…"] * 6] + [fila(f"{kg} ★", pop[kg])]
    tabla = axt.table(cellText=filas, colLabels=["i", "position", "velocity", "cost", "best_position", "best_cost"], loc="center", cellLoc="center")
    tabla.auto_set_font_size(False); tabla.set_fontsize(9.5); tabla.scale(1, 1.5)
    for (r, c), celda in tabla.get_celld().items():
        if r == 0: celda.set_facecolor("#B8D2DA"); celda.set_text_props(weight="bold")
        elif r == len(filas): celda.set_facecolor("#FBF1D6")
    axt.set_title(f"pop: una lista de diccionarios, uno por partícula (las primeras {n_tabla} de {n} y la {kg}, que es gbest ★)")
    plt.tight_layout(); plt.show()


def instantes(H, historial, f, VarMin, VarMax, iteraciones=(0, 3, 10, -1), minimo=(0, 0)):
    """Cuatro instantes del enjambre (H: historia de posiciones, forma (iteraciones + 1, partículas, 2)) y la curva del mejor costo."""
    import matplotlib.pyplot as plt
    X, Y, Z = _malla_funcion(f, (VarMin, VarMax), 200); H = np.asarray(H); historial = np.asarray(historial)
    its = [len(H) - 1 if it < 0 else it for it in iteraciones]
    fig, ejes = plt.subplots(1, len(its) + 1, figsize=(3.6 * len(its) + 4.5, 3.9), gridspec_kw={"width_ratios": [1] * len(its) + [1.35]})
    for ax, it in zip(ejes[:-1], its):
        ax.contourf(X, Y, Z, levels=30, cmap="turbo", alpha=0.85)
        ax.plot(H[it][:, 0], H[it][:, 1], "wo", ms=6, mec="k"); ax.plot(minimo[0], minimo[1], "*", ms=14, mfc="yellow", mec="k")
        ax.set_xlim(VarMin, VarMax); ax.set_ylim(VarMin, VarMax); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"iteración {it} · mejor {historial[it]:.1e}", fontsize=10)
    ejes[-1].semilogy(historial, lw=2, color="#0B5B72"); ejes[-1].set_xlabel("iteración"); ejes[-1].set_ylabel("mejor costo")
    ejes[-1].set_title("curva de convergencia", fontsize=10); ejes[-1].grid(alpha=0.3); plt.tight_layout(); plt.show()


def convergencia(historial, referencia=None, etiqueta="mejor costo del enjambre", etiqueta_ref="referencia", log=False,
                 ylabel="mejor costo", titulo="Curva de convergencia", ylim_rel=None):
    """La curva del mejor costo por iteración; con referencia, la línea del valor exacto (o de la malla). ylim_rel = (abajo, arriba) como fracción de la referencia."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    (ax.semilogy if log else ax.plot)(historial, lw=2, label=etiqueta)
    if referencia is not None:
        ax.axhline(referencia, color="g", ls="--", lw=1.5, label=f"{etiqueta_ref} ({referencia:.2f})")
        if ylim_rel: ax.set_ylim(referencia * ylim_rel[0], referencia * ylim_rel[1])
    ax.set_xlabel("Iteración"); ax.set_ylabel(ylabel); ax.set_title(titulo); ax.legend(); plt.tight_layout(); plt.show()


def resultado_enjambre(historial, pop, gbest, f, VarMin, VarMax, nombre="Sphere"):
    """Lo que devuelve PSO, en dos paneles: la curva de convergencia (log) y las partículas al final sobre la función, con gbest."""
    import matplotlib.pyplot as plt
    X, Y, Z = _malla_funcion(f, (VarMin, VarMax), 150)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    ejes[0].semilogy(historial, lw=2); ejes[0].set_xlabel("Iteración"); ejes[0].set_ylabel("Mejor costo (escala log)"); ejes[0].set_title(f"Curva de convergencia en {nombre}")
    cf = ejes[1].contourf(X, Y, Z, levels=30, cmap="turbo"); fig.colorbar(cf, ax=ejes[1], label=f"{nombre}(x)")
    finales = np.array([p["position"] for p in pop])
    ejes[1].plot(finales[:, 0], finales[:, 1], "wo", ms=4, mec="k", label="partículas al final")
    ejes[1].plot(gbest["position"][0], gbest["position"][1], "*", ms=18, mfc="yellow", mec="k", label="mejor global")
    ejes[1].set_xlabel("$x_1$"); ejes[1].set_ylabel("$x_2$"); ejes[1].legend(loc="upper right"); ejes[1].set_title("Dónde terminaron las partículas")
    plt.tight_layout(); plt.show()


def estadistica_corridas(resultados, etiqueta="Ackley 10-D\n10 semillas", umbral=0.1, optimo=0.0):
    """Diagrama de caja de los mejores costos de varias corridas (semillas), con la línea del óptimo global."""
    import matplotlib.pyplot as plt
    resultados = np.asarray(resultados)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(resultados, tick_labels=[etiqueta])
    ax.scatter(np.ones_like(resultados), resultados, alpha=0.6, zorder=3)
    ax.axhline(optimo, color="g", ls="--", lw=1, label=f"óptimo global (f = {optimo:g})")
    ax.set_ylabel("Mejor costo encontrado"); ax.set_title("Reporta la estadística, no una corrida"); ax.legend(); plt.tight_layout(); plt.show()


def convergencias(historiales, titulo="Curvas de convergencia", ylabel="Mejor costo", log=False):
    """Varias curvas en un eje: historiales = {etiqueta: lista del mejor costo por iteración}."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nombre, h in historiales.items():
        (ax.semilogy if log else ax.plot)(h, lw=2, label=nombre)
    ax.set_xlabel("Iteración"); ax.set_ylabel(ylabel)
    if not log: ax.set_ylim(bottom=0)
    ax.set_title(titulo); ax.legend(); plt.tight_layout(); plt.show()


def paisajes_castigo(alpha, alpha_h, p3, s_max, s_hum, p_exacto, Ms=(50, 10_000), p_max=1000):
    """Tres paisajes de la calificación de Fritz en el plano (p1, p2) con p3 fijo: el cambio de signo del original y el castigo graduado con cada M.
    La línea blanca es la región factible; la cruz, la mejor calificación de cada paisaje; la estrella, el óptimo exacto."""
    import matplotlib.pyplot as plt
    r = np.linspace(0, p_max, 201); P1, P2 = np.meshgrid(r, r)
    S = alpha[:, 0, None, None] * P1 + alpha[:, 1, None, None] * P2 + alpha[:, 2, None, None] * p3
    SH = alpha_h[0] * P1 + alpha_h[1] * P2 + alpha_h[2] * p3
    exceso = np.maximum(S - s_max, 0).sum(axis=0) + np.maximum(SH - s_hum, 0); total = P1 + P2 + p3
    paisajes = [("el original: cambio de signo", np.where(exceso > 0, -total, total))] + [(f"graduado con M = {M:,}".replace(",", " "), total - M * exceso) for M in Ms]
    fig, ejes = plt.subplots(1, len(paisajes), figsize=(6 * len(paisajes), 5))
    for ax, (titulo, calif) in zip(ejes, paisajes):
        im = ax.contourf(P1, P2, calif, levels=np.linspace(-1500, 2300, 30), cmap="viridis", extend="both")
        ax.contour(P1, P2, exceso, levels=[1e-9], colors="w", linewidths=2)
        k = np.unravel_index(np.argmax(calif), calif.shape)
        ax.plot(P1[k], P2[k], "rX", ms=14, mec="w", label=f"mejor calificación ({P1[k]:.0f}, {P2[k]:.0f})")
        ax.plot(p_exacto[0], p_exacto[1], "*", ms=18, mfc="yellow", mec="k", label="óptimo exacto")
        ax.set_title(titulo); ax.set_xlabel("$p_1$ [m³/d]"); ax.set_ylabel("$p_2$ [m³/d]"); ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ejes, label="calificación", shrink=0.8); plt.show()


def curva_beneficio(precio, alpha, c_qp, gslev, h0, pozo=0, p_max=500):
    """Un solo pozo bombeando de 0 a p_max: ingreso (recta), costo (crece con el abatimiento) y beneficio con su máximo."""
    import matplotlib.pyplot as plt
    q = np.linspace(0, p_max, 201); X = np.zeros((len(q), 3)); X[:, pozo] = q
    ingreso = precio * q; costo = np.array([np.sum(c_qp * x * (gslev - h0 + alpha @ x)) for x in X]); benef = ingreso - costo; k = int(np.argmax(benef))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(q, ingreso, lw=2, color="#1F7A55", label="ingreso: precio × bombeo (recta)")
    ax.plot(q, costo, lw=2, color="#B03A2E", label="costo de bombeo (crece con el abatimiento)")
    ax.plot(q, benef, lw=3, color="#0B5B72", label="beneficio = ingreso − costo")
    ax.plot(q[k], benef[k], "o", ms=10, mfc="yellow", mec="k", label=f"máximo: {q[k]:.0f} m³/d, {benef[k]:.2f} $/d")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel(f"bombeo del pozo {pozo + 1} [m³/d] (los otros apagados)"); ax.set_ylabel("$/d")
    ax.set_title("Por qué el óptimo deja la esquina: el beneficio tiene su máximo adentro"); ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()


def paisaje_beneficio(alpha, alpha_h, precio, c_qp, gslev, h0, s_max, s_hum, x_ref, p_max=500):
    """El beneficio en el plano (p1, p2) con el pozo 3 apagado, la región factible (línea blanca) y el mejor punto de la malla (estrella)."""
    import matplotlib.pyplot as plt
    r = np.linspace(0, p_max, 201); P1, P2 = np.meshgrid(r, r)
    XX = np.stack([P1.ravel(), P2.ravel(), np.zeros(P1.size)], axis=1); SS = XX @ alpha.T
    BB = (precio * XX.sum(1) - (c_qp * XX * (gslev - h0 + SS)).sum(1)).reshape(P1.shape)
    ok = ((SS <= s_max).all(axis=1) & (XX @ alpha_h <= s_hum)).reshape(P1.shape)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.contourf(P1, P2, BB, levels=30, cmap="viridis"); fig.colorbar(im, ax=ax, label="beneficio [$/d]")
    ax.contour(P1, P2, ok.astype(float), levels=[0.5], colors="w", linewidths=2)
    ax.plot(x_ref[0], x_ref[1], "*", ms=20, mfc="yellow", mec="k", label=f"mejor de la malla ({x_ref[0]:.0f}, {x_ref[1]:.0f})")
    ax.set_xlabel("$p_1$ [m³/d]"); ax.set_ylabel("$p_2$ [m³/d]"); ax.legend(loc="upper right")
    ax.set_title("El beneficio (pozo 3 apagado): la cima está DENTRO de la región factible"); plt.tight_layout(); plt.show()


def optimo_contra_precio(precios, total_lineal, total_no_lineal, precio_actual=None):
    """El bombeo total óptimo contra el precio del agua: el lineal salta de esquina en esquina; el no lineal crece suave."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.step(precios, total_lineal, where="mid", lw=2.5, color="#B03A2E", label="costo lineal (linprog): salta de esquina en esquina")
    ax.plot(precios, total_no_lineal, "o-", lw=2.5, color="#003D79", label="costo con abatimiento (no lineal): crece suave")
    if precio_actual is not None: ax.axvline(precio_actual, color="#C08A0B", ls=":", label=f"precio de la sección 3 ({precio_actual})")
    ax.set_xlabel("precio del agua [$/m³]"); ax.set_ylabel("bombeo total óptimo [m³/d]"); ax.legend(fontsize=9)
    ax.set_title("Lineal contra no lineal: cómo responde el óptimo al precio"); plt.tight_layout(); plt.show()


def mapas_geologias(geologias=(None, 1, 3), vmin=-8.5, vmax=-4.5):
    """Los mapas de log K de varias geologías, lado a lado y con la misma escala."""
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, len(geologias), figsize=(5.4 * len(geologias), 4.4))
    for ax, geo in zip(np.atleast_1d(ejes), geologias):
        mapa_geologia(geo, ax=ax, vmin=vmin, vmax=vmax)
    plt.tight_layout(); plt.show()


def animar_enjambre(historia, ruta_gif, f=None, lim=(-5, 5), region=None, titulo="El enjambre", fps=8, minimo=(0, 0)):
    """Guarda un GIF con el enjambre iteración por iteración y lo devuelve para mostrarlo en el cuaderno.
    historia: lista o arreglo (iteraciones + 1, partículas, 2). Fondo: una función de prueba (f, lim) o una región factible
    (region = dict(A=..., b=..., nombres=[...], p_optimo=...))."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from IPython.display import Image
    os.makedirs(os.path.dirname(ruta_gif) or ".", exist_ok=True)
    historia = np.asarray(historia); n = len(historia) - 1
    if region is not None:
        ax = graficar_region_factible(region["A"], region["b"], region["nombres"], p_optimo=region.get("p_optimo"), titulo=titulo); fig = ax.figure
        puntos, = ax.plot([], [], "o", ms=6, mfc="w", mec="k", zorder=6)
    else:
        X, Y, Z = _malla_funcion(f, lim, 200)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.contourf(X, Y, Z, levels=30, cmap="turbo", alpha=0.85)
        puntos, = ax.plot([], [], "wo", ms=6, mec="k"); ax.plot(minimo[0], minimo[1], "*", ms=18, mfc="yellow", mec="k")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

    def cuadro(k):
        puntos.set_data(historia[k][:, 0], historia[k][:, 1]); ax.set_title(f"{titulo} · iteración {k} de {n}")
        return puntos,
    anim = FuncAnimation(fig, cuadro, frames=len(historia), interval=1000 // fps)
    anim.save(ruta_gif, writer=PillowWriter(fps=fps), dpi=80); plt.close(fig)
    return Image(ruta_gif)


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
