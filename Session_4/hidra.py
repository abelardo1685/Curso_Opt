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
    # >>> respaldo de la máquina del instructor (el empaquetador lo quita)
    for _cand in (os.path.join(_aqui, "..", "Información", "Versiones_anteriores", "Curso_V2"), os.path.join(_aqui, "..", "Curso_V2")):
        if os.path.isdir(os.path.join(_cand, "rrwpt")):
            sys.path.insert(0, os.path.abspath(_cand))
            break
    # <<< respaldo de la máquina del instructor
    try:
        import rrwpt  # noqa: F401
    except ImportError:
        raise ImportError(
            f"No se encontró rrwpt. Este cuaderno corre con {sys.executable}.\n"
            "Elige el kernel .venv del curso (arriba a la derecha del cuaderno, «Select Kernel» → «Python Environments» → .venv)\n"
            "o vuelve a instalar la paquetería con la guía de instalación (sección 7).") from None

import numpy as np
import warnings
# 2026-11-09: kaleido (exportar figuras a imagen; el curso no lo usa) avisa al importarse si no coincide con la versión de plotly.
# En el entorno del curso no está instalado; en otros Python el aviso solo confunde: se silencia.
warnings.filterwarnings("ignore", message=r"(?s).*Kaleido", category=UserWarning)

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


def _zona(zona, grid):
    """Tendencia en ln K que se suma a cada realización: una zona menos (o más) permeable centrada en (x, y) [m], de radio r [m] y
    amplitud delta (campana gaussiana: delta·exp(−(d/r)²)). zona = (x, y, r, delta). Sesión 4 (2026-11-12): la zona alrededor del pozo 2
    que el homogéneo equivalente no ve, para que el frente optimizado con las geologías quede entre el determinista y el evaluado."""
    x0, y0, r, delta = zona
    n_el = tuple(int(v) for v in grid.data.n_el); d = grid.data.d_pts
    yy, xx = np.meshgrid((np.arange(n_el[0]) + 0.5) * d[0], (np.arange(n_el[1]) + 0.5) * d[1], indexing="ij")
    return (delta * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / r ** 2)).reshape(-1, order="F")


def _generar_ensamble(tipo, k0, k1, verbose=True, zona=None):
    """Las geologías k0, …, k1 − 1 del ensamble `tipo`, con el simulador (una factorización por geología). zona: ver _zona."""
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
        yk = np.asarray(_geostat.uncsim_kfield(ctrl, e["model"], grid, rng=np.random.default_rng(_semilla_ensamble(tipo, k))), dtype=float).reshape(-1)
        if zona is not None:
            yk = yk + _zona(zona, grid)
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


def ensamble(n=200, tipo="optimizacion", verbose=True, zona=None):
    """Un ensamble grande de geologías (Sesión 4): n campos de ln K con el mismo modelo geoestadístico de las diez del curso.

    zona = (x, y, r, delta) añade a cada realización una zona menos permeable (delta < 0 en ln K) centrada en (x, y) m de radio r m
    (Sesión 4, 2026-11-12: la zona alrededor del pozo 2 que el homogéneo no ve); el caché lleva la zona en el nombre.
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
    etiqueta = tipo if zona is None else tipo + "_zona" + "_".join(f"{v:g}" for v in zona)    # p. ej. optimizacion_zona735_450_300_-1
    ruta = os.path.join(carpeta, f"ensamble_{etiqueta}_{n}.npz")
    if os.path.exists(ruta):
        d = np.load(ruta)
        return {k: d[k] for k in d.files}
    guardados = sorted((int(re.search(r"_(\d+)\.npz$", f).group(1)), f) for f in glob.glob(os.path.join(carpeta, f"ensamble_{etiqueta}_*.npz")))
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
    nuevo = _generar_ensamble(tipo, k0, n, verbose=verbose, zona=zona)
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


def superficie_3d(f, lim=(-5, 5), nombre="Sphere", minimo=(0, 0), n=80, alto=560):
    """La función de prueba en 3-D, interactiva (plotly): se gira, se acerca y se mueve con el mouse. Marca el mínimo global con una estrella.
    Sirve para ver la forma completa antes de optimizar: Sphere es un solo valle (convexa); Ackley, un embudo lleno de hoyos."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as e:
        raise ImportError("Falta plotly (la figura 3-D interactiva):  pip install -r requirements.txt") from e
    # el mismo renderer que frente_3d: el que dibujan VS Code y JupyterLab, también al reabrir el cuaderno
    if "google.colab" not in sys.modules:
        pio.renderers.default = "plotly_mimetype"
    X, Y, Z = _malla_funcion(f, lim, n)
    z_min = float(f(np.array(minimo, dtype=float)))
    fig = go.Figure(go.Surface(x=X, y=Y, z=Z, colorscale="Turbo", colorbar=dict(title=f"{nombre}(x)"), opacity=0.95,
                               contours=dict(z=dict(show=True, usecolormap=True, project_z=True))))
    fig.add_trace(go.Scatter3d(x=[minimo[0]], y=[minimo[1]], z=[z_min], mode="markers+text", text=[f"mínimo global {tuple(minimo)}"],
                               textposition="top center", marker=dict(size=7, color="gold", symbol="diamond", line=dict(color="black", width=2)),
                               showlegend=False))
    fig.update_layout(title=f"{nombre} en 3-D · gírala con el mouse", height=alto, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title=f"{nombre}(x)", camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0))))
    fig.show()


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


def registrar_pso(problem, MaxIter=100, PopSize=100, c1=1.5, c2=2.0, w=1.0, wdamp=0.995):
    """La misma corrida que pso.PSO (el mismo ciclo y el mismo orden de números al azar: con la misma semilla da el mismo resultado), pero
    guardando dónde está cada partícula en cada iteración, para animarla. Devuelve (H, H_costo, H_g): posiciones (MaxIter + 1, PopSize, nVar),
    el mejor costo y la mejor posición del grupo en cada iteración. Fijar la semilla antes (np.random.seed) igual que con pso.PSO."""
    f, lo, hi, n = problem["CostFunction"], problem["VarMin"], problem["VarMax"], problem["nVar"]
    rep = problem.get("ResFunction"); minimiza = problem.get("MinProblem", 1) == 1
    mejor = (lambda a, b: a < b) if minimiza else (lambda a, b: a > b)
    pos = np.zeros((PopSize, n)); vel = np.zeros((PopSize, n)); pb = np.zeros((PopSize, n)); pbc = np.zeros(PopSize)
    g, gc = None, (np.inf if minimiza else -np.inf)
    for i in range(PopSize):
        pos[i] = np.random.uniform(lo, hi, n)
        if rep is not None: pos[i] = rep(pos[i])
        c = f(pos[i]); pb[i] = pos[i].copy(); pbc[i] = c
        if mejor(c, gc): g, gc = pos[i].copy(), c
    H, Hc, Hg = [pos.copy()], [gc], [g.copy()]
    for _ in range(MaxIter):
        for i in range(PopSize):
            vel[i] = w * vel[i] + c1 * np.random.rand(n) * (pb[i] - pos[i]) + c2 * np.random.rand(n) * (g - pos[i])
            pos[i] = np.minimum(np.maximum(pos[i] + vel[i], lo), hi)
            if rep is not None: pos[i] = rep(pos[i])
            c = f(pos[i])
            if mejor(c, pbc[i]):
                pb[i], pbc[i] = pos[i].copy(), c
                if mejor(c, gc): g, gc = pos[i].copy(), c
        w *= wdamp; H.append(pos.copy()); Hc.append(gc); Hg.append(g.copy())
    return np.array(H), np.array(Hc), np.array(Hg)


def animar_enjambre_interactivo(H, H_costo, H_g, f, lim=(-5, 5), nombre="Sphere", minimo=(0, 0), alto=560):
    """El enjambre en movimiento, en dos vistas lado a lado: desde arriba (curvas de nivel) y en 3-D sobre la superficie (que se puede girar
    mientras corre). Botón ▶ y barra para ir a cualquier iteración (plotly). H, H_costo, H_g: lo que devuelve registrar_pso."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError("Falta plotly (la figura interactiva):  pip install -r requirements.txt") from e
    if "google.colab" not in sys.modules:
        pio.renderers.default = "plotly_mimetype"
    X, Y, Z = _malla_funcion(f, lim, 60)
    zf = lambda P: np.array([f(p) for p in P])
    n = len(H) - 1; zmin = float(f(np.array(minimo, dtype=float)))
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "scene"}]], column_widths=[0.45, 0.55],
                        subplot_titles=("desde arriba", "en 3-D (gírala con el mouse)"))
    fig.add_trace(go.Contour(x=X[0], y=Y[:, 0], z=Z, colorscale="Turbo", showscale=False, ncontours=25, opacity=0.85), 1, 1)
    fig.add_trace(go.Scatter(x=H[0][:, 0], y=H[0][:, 1], mode="markers", marker=dict(size=7, color="white", line=dict(color="black", width=1)),
                             name="partículas"), 1, 1)
    fig.add_trace(go.Scatter(x=[H_g[0][0]], y=[H_g[0][1]], mode="markers", marker=dict(size=13, color="gold", symbol="diamond",
                             line=dict(color="black", width=1.5)), name="mejor del grupo (g)"), 1, 1)
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale="Turbo", showscale=False, opacity=0.7), 1, 2)
    fig.add_trace(go.Scatter3d(x=H[0][:, 0], y=H[0][:, 1], z=zf(H[0]), mode="markers", marker=dict(size=4, color="white",
                               line=dict(color="black", width=1)), showlegend=False), 1, 2)
    fig.add_trace(go.Scatter3d(x=[H_g[0][0]], y=[H_g[0][1]], z=[float(f(H_g[0]))], mode="markers", marker=dict(size=7, color="gold",
                               symbol="diamond"), showlegend=False), 1, 2)
    titulo = lambda k: f"{nombre}: el enjambre en la iteración {k} de {n} · mejor costo {H_costo[k]:.2e}"
    fig.frames = [go.Frame(name=str(k), traces=[1, 2, 4, 5], layout=dict(title_text=titulo(k)),
                           data=[go.Scatter(x=H[k][:, 0], y=H[k][:, 1]), go.Scatter(x=[H_g[k][0]], y=[H_g[k][1]]),
                                 go.Scatter3d(x=H[k][:, 0], y=H[k][:, 1], z=zf(H[k])),
                                 go.Scatter3d(x=[H_g[k][0]], y=[H_g[k][1]], z=[float(f(H_g[k]))])]) for k in range(n + 1)]
    paso = dict(frame=dict(duration=250, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")
    fig.update_layout(title_text=titulo(0), height=alto, margin=dict(l=10, r=10, t=70, b=10),
                      xaxis=dict(range=list(lim), title="x₁", scaleanchor="y"), yaxis=dict(range=list(lim), title="x₂"),
                      scene=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title=f"{nombre}(x)", camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0))),
                      legend=dict(orientation="h", y=-0.08, x=0.0),
                      updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.13, showactive=False,
                                        buttons=[dict(label="▶ correr", method="animate", args=[None, paso]),
                                                 dict(label="❚❚ pausa", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
                      sliders=[dict(active=0, y=0, x=0.1, len=0.9, currentvalue=dict(prefix="iteración "),
                                    steps=[dict(label=str(k), method="animate", args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                                           for k in range(n + 1)])])
    fig.show()


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



# ----------------------------------------------------------------- gráficas multiobjetivo (sesión 3; 2026-11-11: el cuaderno solo llama, el algoritmo queda a la vista)
def frente_verdadero_dtlz3(n=35):
    """El frente verdadero de la DTLZ3 de pyOMOPSO: puntos sobre el octante de la esfera de radio 1 (para verlo antes de correr)."""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    u, v = u.ravel() * np.pi / 2, v.ravel() * np.pi / 2
    return np.column_stack([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), np.sin(u)])


def frente_angulos(F, ejes=("f1", "f2", "f3"), titulo=None, azimuts=(0, 30, 75, 120), elev=20, referencia=None, punto=None, color="green"):
    """Un frente de tres objetivos visto desde cuatro azimuts (2 × 2). referencia: otro frente en gris; punto: una solución en rojo."""
    import matplotlib.pyplot as plt
    F = np.asarray(F)
    fig = plt.figure(figsize=(10, 8))
    for i, azim in enumerate(azimuts):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        if referencia is not None:
            R = np.asarray(referencia); ax.scatter3D(R[:, 0], R[:, 1], R[:, 2], color="#c9d6e4", s=2, alpha=0.4, label="referencia (malla)")
        ax.scatter3D(F[:, 0], F[:, 1], F[:, 2], color=color, edgecolors="black", linewidths=0.5, label="frente de Pareto")
        if punto is not None:
            ax.scatter3D(punto[0], punto[1], punto[2], color="red", s=60, label="más cercana al origen")
        ax.set_xlabel(ejes[0]); ax.set_ylabel(ejes[1]); ax.set_zlabel(ejes[2]); ax.set_title(f"Azimut = {azim}°"); ax.view_init(elev=elev, azim=azim)
        if i == 0: ax.legend(fontsize=8, loc="upper left")
    if titulo: fig.suptitle(titulo)
    plt.tight_layout(); plt.show()


def histograma_radios(radios, verdadero=1.0, titulo="¿Sobre la esfera?"):
    """Histograma de los radios de las soluciones de DTLZ3 y la línea del frente verdadero."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 3.6))
    plt.hist(radios, bins=30, color="#2E7DB5", edgecolor="k", alpha=0.8); plt.axvline(verdadero, color="#B03A2E", ls="--", lw=2, label="frente verdadero")
    plt.xlabel("radio $\\sqrt{f_1^2+f_2^2+f_3^2}$"); plt.ylabel("soluciones"); plt.legend(); plt.title(titulo); plt.tight_layout(); plt.show()


def frente_2d(*curvas, punto=None, xl="costo [$/d]", yl="índice de necesidad no cubierta", titulo=None, ax=None):
    """Frentes de dos objetivos. Cada curva es (F, etiqueta) en puntos, o (F, etiqueta, 'linea') como línea; un cuarto elemento fija el color.
    punto = (f, etiqueta) marca una solución en rojo."""
    import matplotlib.pyplot as plt
    propio = ax is None
    if propio: fig, ax = plt.subplots(figsize=(7, 4.6))
    col_l = ["#003D79", "#C08A0B", "#1F7A55", "#6A3D9A"]; col_p = ["#C08A0B", "#2E7DB5", "#1F7A55"]; nl = npt = 0
    for c in curvas:
        F, et = np.asarray(c[0]), c[1]; estilo = c[2] if len(c) > 2 else "puntos"; col = c[3] if len(c) > 3 else None
        if estilo == "linea":
            ax.plot(F[:, 0], F[:, 1], "-", color=col or col_l[nl % 4], lw=2.5, label=et); nl += 1
        else:
            ax.plot(F[:, 0], F[:, 1], "o", mfc=col or col_p[npt % 3], mec="k", ms=6, label=et); npt += 1
    if punto is not None: ax.plot(punto[0][0], punto[0][1], "o", color="red", ms=10, label=punto[1])
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=9)
    if titulo: ax.set_title(titulo)
    if propio: plt.tight_layout(); plt.show()


def mapa_indice(c_base, a_idx, k, p3=100, p_max=500):
    """El índice de necesidad no cubierta en función de los bombeos de los pozos 1 y 2, con el pozo 3 fijo (mapa de calor del original)."""
    import matplotlib.pyplot as plt
    P1, P2 = np.meshgrid(np.linspace(0, p_max, 100), np.linspace(0, p_max, 100))
    Q = np.exp(-k * (c_base[0] + a_idx[0] * P1)) + np.exp(-k * (c_base[1] + a_idx[1] * P2)) + np.exp(-k * (c_base[2] + a_idx[2] * p3))
    plt.figure(figsize=(7.5, 5.5))
    plt.colorbar(plt.contourf(P1, P2, Q, levels=50, cmap="viridis"), label="índice de necesidad no cubierta I")
    plt.xlabel("bombeo del pozo 1, $p_1$ [m³/d]"); plt.ylabel("bombeo del pozo 2, $p_2$ [m³/d]")
    plt.title(f"El índice en función de los bombeos (pozo 3 fijo en {p3} m³/d)"); plt.tight_layout(); plt.show()


def frente_referencia(evaluar, paso=10, p_max=500, s_max=3.0, s_hum=1.25, **kw):
    """El frente de referencia por fuerza bruta (sesión 3, sección 2.3; 2026-11-16, ARP: «que esté en hidra y el cuaderno declare lo mínimo»).
    Prueba TODAS las decisiones de una malla (cada `paso` m³/d por pozo, de 0 a `p_max`) con la función `evaluar(p)` del cuaderno, que devuelve
    un diccionario con 's' (abatimientos en los pozos), 'humedal', 'costo' e 'indice'; los `kw` se le pasan tal cual (por ejemplo la matriz de
    otra geología). Devuelve (X_leg, F_leg, X_ref, F_ref): las decisiones legales con sus objetivos (costo, índice) y, de ellas, las no dominadas."""
    g = np.arange(0, p_max + 1, paso)
    X = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    S = np.empty((len(X), 3)); H = np.empty(len(X)); F = np.empty((len(X), 2))
    for i, p in enumerate(X):
        e = evaluar(p, **kw)
        S[i], H[i], F[i] = e["s"], e["humedal"], (e["costo"], e["indice"])
    legal = (S <= s_max + 1e-12).all(axis=1) & (H <= s_hum + 1e-12)
    X_leg, F_leg = X[legal], F[legal]
    orden = np.lexsort((F_leg[:, 1], F_leg[:, 0]))          # de la más barata a la más cara (y por índice en los empates)
    k, mejor = [], np.inf
    for i in orden:
        if F_leg[i, 1] < mejor:                             # no dominada: menos índice que todas las más baratas
            k.append(i); mejor = F_leg[i, 1]
    return X_leg, F_leg, X_leg[k], F_leg[k]


def espacios(X_leg, F_leg, X_ref, F_ref, n_muestra=4000, xl="costo [$/d]  (menos es mejor)", yl="índice de necesidad no cubierta  (menos es mejor)"):
    """Los dos espacios lado a lado: las decisiones legales (gris) y las no dominadas (oro) en decisión (3-D) y en objetivos (2-D)."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 5))
    muestra = np.random.default_rng(0).choice(len(X_leg), min(n_muestra, len(X_leg)), replace=False)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(X_leg[muestra, 0], X_leg[muestra, 1], X_leg[muestra, 2], s=2, c="#c9d6e4")
    ax1.scatter(X_ref[:, 0], X_ref[:, 1], X_ref[:, 2], s=10, c="#C08A0B")
    ax1.set_xlabel("$p_1$"); ax1.set_ylabel("$p_2$"); ax1.set_zlabel("$p_3$"); ax1.set_title("Espacio de DECISIÓN (oro: conjunto de Pareto)"); ax1.view_init(22, 40)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(F_leg[muestra, 0], F_leg[muestra, 1], ".", color="#c9d6e4", ms=3, label="decisiones legales")
    ax2.plot(F_ref[:, 0], F_ref[:, 1], "-", color="#C08A0B", lw=3, label="frente de Pareto (referencia)")
    ax2.set_xlabel(xl); ax2.set_ylabel(yl); ax2.set_title("Espacio de OBJETIVOS"); ax2.legend()
    plt.tight_layout(); plt.show()


_REFERENCIA_3 = {}                                            # memoria de la superficie de referencia: se calcula una sola vez por `evaluar`


def _referencia_3(ref):
    """La superficie de referencia de tres objetivos. `ref` puede ser la tabla ya calculada o la función `evaluar` del cuaderno; en ese caso
    se calcula por dentro (una sola vez: queda en memoria) con la malla cada 20 m³/d."""
    return superficie_referencia(ref, imprimir=False)[0] if callable(ref) else ref


def superficie_referencia(evaluar, paso=20, p_max=500, s_max=3.0, s_hum=1.25, imprimir=True):
    """La superficie de referencia de tres objetivos (sesión 3, sección 3.2), por fuerza bruta: prueba todas las decisiones de una malla (cada `paso`
    m³/d por pozo) con `evaluar`, se queda con las legales y, de ellas, con las no dominadas en (abatimiento total, costo, índice). Imprime cuántas
    hay y las correlaciones entre los objetivos sobre el frente. Devuelve (F_ref, X_ref). 2026-11-16, ARP: la celda solo lleva el método."""
    clave = (paso, p_max, s_max, s_hum)
    if _REFERENCIA_3.get(clave, (None,))[0] is evaluar and not imprimir:
        return _REFERENCIA_3[clave][1:]
    g = np.arange(0, p_max + 1, float(paso))
    X = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    ev = [evaluar(p) for p in X]
    legal = np.array([(e["s"] <= s_max).all() and e["humedal"] <= s_hum for e in ev])
    X = X[legal]
    F = np.array([(e["s"].sum(), e["costo"], e["indice"]) for e, ok in zip(ev, legal) if ok])
    nd = np.ones(len(F), bool)
    for i in range(len(F)):                                 # no dominada: ninguna otra es igual o mejor en todo y mejor en algo
        if nd[i] and ((F <= F[i]).all(axis=1) & (F < F[i]).any(axis=1)).any():
            nd[i] = False
    F_ref, X_ref = F[nd], X[nd]
    _REFERENCIA_3[clave] = (evaluar, F_ref, X_ref)
    if imprimir:
        r = np.corrcoef(F_ref.T)
        print(f"{len(F)} decisiones legales; {len(F_ref)} no dominadas: la superficie de referencia")
        print(f"correlaciones sobre el frente: abatimiento-costo {r[0, 1]:.2f} | abatimiento-índice {r[0, 2]:.2f} | costo-índice {r[1, 2]:.2f}")
    return F_ref, X_ref


def dominancia_por_objetivos(evaluar, paso=20, p_max=500, s_max=3.0, s_hum=1.25):
    """Cuántas decisiones legales de la malla (cada `paso` m³/d) quedan sin dominar según cuántos objetivos se cuenten: dos (costo, índice),
    tres (más el abatimiento total) y cuatro (más el abatimiento del humedal). Enseña que cada objetivo que se añade debilita la dominancia:
    más decisiones «empatan» y el frente crece (sesión 3, «¿Y un cuarto objetivo?»)."""
    g = np.arange(0, p_max + 1, float(paso))
    X = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    ev = [evaluar(p) for p in X]
    F = np.array([(e["costo"], e["indice"], e["s"].sum(), e["humedal"]) for e in ev if (e["s"] <= s_max).all() and e["humedal"] <= s_hum])
    print(f"de {len(F)} decisiones legales de la malla, quedan sin dominar:")
    for n, nombre in ((2, "con 2 objetivos (costo, índice)"), (3, "con 3 objetivos (+ abatimiento total)"), (4, "con 4 objetivos (+ abatimiento del humedal)")):
        Fk = F[:, :n]
        nd = sum(not ((Fk <= Fk[i]).all(axis=1) & (Fk < Fk[i]).any(axis=1)).any() for i in range(len(Fk)))
        print(f"  {nombre:44s} {nd:5d}   ({100 * nd / len(Fk):.1f} %)")


def calidad_frente(F_ref, F):
    """Imprime cuántas soluciones tiene el frente F y su distancia media al frente de referencia (en % del tamaño de la referencia).
    Con tres objetivos, F_ref puede ser la función `evaluar` del cuaderno: la referencia (las no dominadas de una malla) se calcula por dentro."""
    F_ref = _referencia_3(F_ref)
    print(f"{len(F)} soluciones; distancia media al frente de referencia: {distancia_a_referencia(F_ref, F) * 100:.2f} % del tamaño del frente")


def _normalizar_tres(F, F_ref):
    """Normaliza el frente y la referencia con la escala del frente: 0 = lo mejor del frente, 1 = lo peor."""
    lo, hi = F.min(0), F.max(0)
    return (F - lo) / (hi - lo), (F_ref - lo) / (hi - lo)


def frente_tres_objetivos(F, F_ref, k, ejes=("Abatimiento", "Costo", "Índice")):
    """Sesión 3, sección 3.4.2: el frente de OMOPSO (blanco) sobre la referencia (gris), normalizados, con la solución k en rojo:
    cuatro ángulos fijos y la figura interactiva. F_ref puede ser la función `evaluar`: la referencia se calcula por dentro, y se imprime
    cuántas soluciones tiene y la correlación entre abatimiento y costo sobre ella (lo que hace angosta a la cinta)."""
    if callable(F_ref):
        F_ref = _referencia_3(F_ref); r = np.corrcoef(F_ref.T)
        print(f"la referencia (gris): {len(F_ref)} no dominadas de una malla cada 20 m³/d; sobre ella, correlación abatimiento-costo {r[0, 1]:.2f}")
    Fn, Fr = _normalizar_tres(F, F_ref)
    frente_angulos(Fn, ejes=ejes, azimuts=(0, 45, 90, 135), referencia=Fr, punto=Fn[k], color="white",
                   titulo="OMOPSO (blanco) sobre la referencia (gris); en rojo, la más cercana al origen")
    frente_3d((Fr, "Referencia (malla)", "#9fb3c8", 2), (Fn, "Frente de Pareto", "white"), ejes=ejes,
              punto=(Fn[k], "más cercana al origen"), titulo="Gíralo con el mouse; un clic en la leyenda oculta cada capa", alto=650).show()


def caras_tres_objetivos(F, F_ref, k, ejes=("Abatimiento", "Costo", "Índice")):
    """Sesión 3, sección 3.4.3: los tres pares de objetivos en 2-D, normalizados, con la solución k marcada."""
    import matplotlib.pyplot as plt
    Fn, Fr = _normalizar_tres(F, _referencia_3(F_ref))
    caras_frente((Fr, "Referencia (malla)", "#9fb3c8", 2), (Fn, "Frente de Pareto", "white"), ejes=ejes,
                 punto=(Fn[k], "más cercana al origen"), titulo="Las caras del frente (normalizado)")
    plt.show()


def compromiso_tres_objetivos(Fn, k, X, evaluar, p_dos):
    """Sesión 3, sección 3.5: los bombeos y los objetivos del compromiso de tres objetivos (solución k), contra el de dos objetivos
    (p_dos), y su lectura en coordenadas paralelas y en el acuífero."""
    p = X[k]; e = evaluar(p)
    print(f"compromiso de tres objetivos: p = ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) = {p.sum():.0f} m³/d")
    print(f"  abatimiento total {e['s'].sum():.2f} m | costo {e['costo']:.1f} $/d | índice {e['indice']:.3f} | humedal {e['humedal']:.3f} m")
    print(f"compromiso de dos objetivos:  p = ({p_dos[0]:.0f}, {p_dos[1]:.0f}, {p_dos[2]:.0f}) = {p_dos.sum():.0f} m³/d")
    paralelas_acuifero(Fn, ["abatimiento total", "costo", "índice"], k, p)


def castigo_con_numeros(p, evaluar, funcion, Ms=(0, 10, 1000), s_max=3.0, s_hum=1.25):
    """El castigo graduado con números (sesión 3, «Las restricciones dentro de OMOPSO»): para la decisión p imprime cuánto se pasa de las reglas
    (v, en metros) y lo que la multa M·v le suma a cada objetivo de `funcion(p, M=M)` con cada M. 2026-11-16, ARP: la celda solo lleva el método."""
    e = evaluar(p)
    exc_s, exc_h = np.maximum(e["s"] - s_max, 0), max(e["humedal"] - s_hum, 0)
    v = exc_s.sum() + exc_h
    print(f"B: pozos {np.round(e['s'], 4)} m, humedal {e['humedal']:.4f} m -> excesos {np.round(exc_s, 4)} y {exc_h:.4f}: v = {v:.4f} m")
    for M in Ms:
        f = funcion(p, M=M).ravel()
        print(f"M = {M:5d}: costo {e['costo']:.1f} + {M} × {v:.4f} = {f[0]:8.1f} $/d | índice {e['indice']:.3f} + {M} × {v:.4f} = {f[1]:8.3f}")


def comparar_castigos(optimizador, parametros, casos, semillas=(42, 1, 2), n_var=3):
    """Prueba cada planteamiento de `casos` = {nombre: (función, tope de bombeo, M)} con el optimizador, una corrida por semilla, e imprime una
    tabla con cuántas soluciones quedan en el archivo. M = None usa la función tal cual; con un número, la llama con ese M (castigo graduado).
    Pensada para corridas cortas de prueba (sesión 3, «Las restricciones dentro de OMOPSO»)."""
    print(f"{'soluciones en el archivo':44s}  " + "  ".join(f"semilla {s:<3d}" for s in semillas))
    for nombre, (funcion, tope, M) in casos.items():
        f = funcion if M is None else (lambda x, funcion=funcion, M=M: funcion(x, M=float(M)))
        cuenta = []
        for s in semillas:
            np.random.seed(s)
            archivo, _, _ = optimizador({"CostFunction": f, "nVar": n_var, "VarMin": 0, "VarMax": tope}, parametros)
            cuenta.append(len(archivo))
        print(f"{nombre:44s}  " + "  ".join(f"{c:^11d}" for c in cuenta))


def distancia_a_referencia(F_ref, F):
    """Distancia media de cada solución de F a la más cercana del frente de referencia, en fracción del tamaño del frente
    (cada objetivo dividido entre su rango en la referencia). La medida de calidad de la sesión 3, sección 2.5."""
    escala = F_ref.max(0) - F_ref.min(0)
    return float(np.mean([np.min(np.linalg.norm((F_ref - f) / escala, axis=1)) for f in F]))


def frente_contra_referencia(F_ref, X_ref, F, X, xl="costo [$/d]", yl="índice de necesidad no cubierta"):
    """OMOPSO (oro) contra la referencia de la malla (azul) en el espacio de objetivos y en el de decisión.
    Antes de dibujar imprime cuántas soluciones hay y su distancia media a la referencia (2026-11-16, ARP: la medida sale de la celda)."""
    import matplotlib.pyplot as plt
    calidad_frente(F_ref, F)
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    frente_2d((F_ref, "referencia (malla)", "linea"), (F, "OMOPSO"), xl=xl, yl=yl, titulo="Espacio de objetivos", ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(X_ref[:, 0], X_ref[:, 1], X_ref[:, 2], s=6, c="#003D79", alpha=0.4, label="referencia")
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], s=20, c="#C08A0B", edgecolors="k", label="OMOPSO")
    ax2.set_xlabel("$p_1$"); ax2.set_ylabel("$p_2$"); ax2.set_zlabel("$p_3$"); ax2.view_init(22, 40); ax2.legend(); ax2.set_title("Espacio de decisión")
    plt.tight_layout(); plt.show()


def compromiso_con_pesos(F, X, k, w, evaluar, k_iguales, xl="costo [$/d]", yl="índice de necesidad no cubierta"):
    """El compromiso elegido con pesos w (solución k del frente F, bombeos X) contra el de pesos iguales (k_iguales): imprime los bombeos,
    el costo y el índice de los dos y los marca sobre el frente (sesión 3, Tu turno 3; 2026-11-16, ARP: la celda solo lleva el método)."""
    import matplotlib.pyplot as plt
    for etiqueta, j in ((f"pesos {tuple(w)}", k), ("pesos iguales", k_iguales)):
        p = X[j]; e = evaluar(p)
        print(f"{etiqueta:<15} p = ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) = {p.sum():.0f} m³/d | costo {e['costo']:.1f} $/d | índice {e['indice']:.3f}")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    frente_2d((F, "frente de OMOPSO"), punto=(F[k], f"compromiso con pesos {tuple(w)}"), xl=xl, yl=yl,
              titulo="El compromiso se mueve sobre la misma curva", ax=ax)
    ax.plot(F[k_iguales, 0], F[k_iguales, 1], "s", mfc="none", mec="k", mew=1.8, ms=11, label="compromiso con pesos iguales")
    ax.legend(fontsize=9); plt.tight_layout(); plt.show()


def compromiso_acuifero(F, k, p, e=None, xl="costo [$/d]", yl="índice", titulo="El compromiso sobre el frente", titulo_mapa="La solución de compromiso en el acuífero"):
    """La solución k del frente F (en rojo) y, al lado, su abatimiento en el acuífero (hidra.simular(p)). Si se pasa `e` (lo que devuelve
    `evaluar(p)`), imprime antes los bombeos, el costo, el índice y los abatimientos del compromiso (2026-11-16, ARP: la celda solo lleva el método)."""
    import matplotlib.pyplot as plt
    if e is not None:
        print(f"compromiso de dos objetivos: p = {np.round(p, 1)} m³/d (total {p.sum():.0f})")
        print(f"  costo {e['costo']:.1f} $/d | índice {e['indice']:.3f} | humedal {e['humedal']:.3f} m | pozos {np.round(e['s'], 2)} m")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.plot(F[:, 0], F[:, 1], "o", mfc="white", mec="k", ms=6, label="frente de OMOPSO")
    ax1.plot(F[k, 0], F[k, 1], "o", color="red", ms=10, label="solución de compromiso")
    ax1.set_xlabel(xl); ax1.set_ylabel(yl); ax1.legend(); ax1.set_title(titulo)
    mapa(simular(p), titulo=titulo_mapa, ax=ax2)
    plt.tight_layout(); plt.show()


def paralelas(Fn, nombres, k=None, ax=None, titulo="Coordenadas paralelas: cada línea es una solución", color="#2E7DB5"):
    """Coordenadas paralelas de un frente normalizado (una línea por solución); k resalta una solución en rojo."""
    import matplotlib.pyplot as plt
    propio = ax is None
    if propio: fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(range(Fn.shape[1]))
    for fila in Fn: ax.plot(xs, fila, color=color, alpha=0.15 if k is None else 0.2)
    if k is not None: ax.plot(xs, Fn[k], color="red", lw=3, label="solución de compromiso"); ax.legend()
    ax.set_xticks(xs); ax.set_xticklabels(nombres); ax.set_ylabel("normalizado (0 = mejor)"); ax.set_title(titulo)
    if propio: plt.tight_layout(); plt.show()


def paralelas_acuifero(Fn, nombres, k, p, titulo_mapa="El compromiso de tres objetivos en el acuífero"):
    """Coordenadas paralelas con el compromiso en rojo y, al lado, su abatimiento en el acuífero."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    paralelas(Fn, nombres, k=k, ax=ax1, titulo="Coordenadas paralelas")
    mapa(simular(p), titulo=titulo_mapa, ax=ax2)
    plt.tight_layout(); plt.show()


def mapas_campo_respuesta(campo=None):
    """Los tres mapas de respuesta unitaria (mm de abatimiento por m³/d): cada columna de la matriz α es uno de ellos leído en los pozos."""
    import matplotlib.pyplot as plt
    if campo is None: campo = campo_respuesta()
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.6))
    for j in range(3):
        im = ejes[j].imshow(campo[:, :, j] * 1000, origin="lower", extent=[0, LADO, 0, LADO], cmap="Blues")
        plt.colorbar(im, ax=ejes[j], label="mm de abatimiento por m³/d", shrink=0.85)
        for k, (xx, yy) in enumerate(pozos_xy()):
            ejes[j].plot(xx, yy, "o", ms=8, mfc="#C08A0B" if k == j else "white", mec="k")
        ejes[j].plot(*humedal_xy(), "P", ms=10, mfc="#8fd18f", mec="k")
        ejes[j].set_title(f"columna {j + 1}: solo bombea el pozo {j + 1}")
    plt.tight_layout(); plt.show()


# ----------------------------------------------------------------- gráficas de robustez (sesión 4; 2026-11-12: el cuaderno solo llama, el algoritmo queda a la vista)
_AZUL, _ORO, _ROJO, _VERDE, _GRIS = "#003D79", "#C08A0B", "#B03A2E", "#2e9b4f", "#555555"


def dos_cimas(f, xs, sigma, cimas=(2.0, 7.0)):
    """La función de prueba de la sesión 4 con sus dos cimas y, sobre cada una, el ancho de la incertidumbre (x ± 2σ)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(xs, f(xs), color=_AZUL, lw=2.5, label="f(x): lo que da cada decisión")
    for k, c in enumerate(cimas):
        ax.axvspan(c - 2 * sigma, c + 2 * sigma, color=_ORO, alpha=0.18, label="x ± 2σ: donde puede caer X" if k == 0 else None)
        ax.plot(c, f(c), "*", ms=16, mfc=_ORO, mec="k")
        ax.annotate(f"x = {c:g}, f = {f(c):.2f}", (c, f(c)), xytext=(8, -4), textcoords="offset points", fontsize=10)
    ax.set_xlabel("x (la decisión)"); ax.set_ylabel("f(x)"); ax.set_ylim(0, 1.15); ax.legend(loc="upper right")
    ax.set_title(f"Dos cimas y la incertidumbre (σ = {sigma}): la angosta cabe dentro del error"); fig.tight_layout(); plt.show()


def esperanza_contra_f(xs, fx, Ex, sigma=None):
    """f(x) contra su valor esperado E[f(X)], X ~ N(x, σ), y los dos óptimos: el determinista (el máximo de f) y el robusto (el máximo de E)."""
    import matplotlib.pyplot as plt
    xs, fx, Ex = np.asarray(xs), np.asarray(fx), np.asarray(Ex)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, fx, color=_AZUL, lw=2.5, label="f(x): sin incertidumbre")
    ax.plot(xs, Ex, color=_ORO, lw=2.5, label="E[f(X)]: con incertidumbre" + (f" (σ = {sigma})" if sigma is not None else ""))
    ax.plot(xs[np.argmax(fx)], fx.max(), "*", ms=18, mfc=_AZUL, mec="k", label=f"óptimo determinista: x = {xs[np.argmax(fx)]:.1f}")
    ax.plot(xs[np.argmax(Ex)], Ex.max(), "*", ms=18, mfc=_ORO, mec="k", label=f"óptimo robusto: x = {xs[np.argmax(Ex)]:.1f}")
    ax.set_xlabel("x"); ax.set_ylabel("valor"); ax.legend(); ax.set_title("La cima angosta no resiste la incertidumbre")
    fig.tight_layout(); plt.show()


def error_montecarlo(N, curvas, ylabel="E[f(X)] estimada", xlabel="número de muestras N"):
    """La media acumulada con las primeras N muestras y su banda de ± 2 errores estándar. curvas: {etiqueta: (media_acumulada, error)}."""
    import matplotlib.pyplot as plt
    colores = [_AZUL, _ORO, _ROJO, _VERDE]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for k, (etiqueta, (media, error)) in enumerate(curvas.items()):
        c = colores[k % len(colores)]
        ax.fill_between(N, media - 2 * error, media + 2 * error, color=c, alpha=0.2)
        ax.plot(N, media, color=c, lw=2, label=f"x = {etiqueta}" if not isinstance(etiqueta, str) else etiqueta)
    ax.set_xscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend()
    ax.set_title("La estimación se estabiliza como 1/√N (la banda: ± 2 errores estándar)"); fig.tight_layout(); plt.show()


def medidas_robustez(xs, medidas):
    """Cada medida de robustez (media, peor caso, fiabilidad…) contra la decisión x, con la x donde cada una es mejor. medidas: {nombre: valores}."""
    import matplotlib.pyplot as plt
    xs = np.asarray(xs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nombre, m in medidas.items():
        m = np.asarray(m); ax.plot(xs, m, lw=2, label=f"{nombre}: mejor en x = {xs[np.argmax(m)]:.1f}")
    ax.set_xlabel("x"); ax.legend(fontsize=9); ax.set_title("Cada medida elige su decisión"); fig.tight_layout(); plt.show()


def menu_robusto(xs, Ex, X_ce, F_ce):
    """El menú de dos objetivos del ejemplo de las dos cimas: el costo de la decisión (x) contra lo que se espera (E[f]); OMOPSO sobre la curva de todas las decisiones."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, Ex, color="#c9d6e4", lw=3, label="E[f(X)] de todas las decisiones")
    ax.plot(X_ce[:, 0], 1.5 - F_ce[:, 1], "o", mfc=_ORO, mec="k", ms=6, label=f"el frente de OMOPSO ({len(F_ce)} soluciones)")
    ax.set_xlabel("costo de la decisión (x)"); ax.set_ylabel("lo que se espera, E[f]"); ax.legend(loc="upper left")
    ax.set_title("El menú robusto: dos tramos"); fig.tight_layout(); plt.show()


def costo_robustez_miniatura(sigmas, E_rob, x_rob, promesa=1.0):
    """Lo que se puede esperar del óptimo robusto contra la incertidumbre σ, y la promesa del modelo sin incertidumbre."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(sigmas, E_rob, "o-", color=_AZUL, lw=2.5, label="lo que se puede esperar (óptimo robusto)")
    ax.axhline(promesa, color=_ROJO, ls="--", label="lo que promete el modelo sin incertidumbre")
    for s, e, x in zip(sigmas, E_rob, x_rob):
        ax.annotate(f"x = {x:.1f}", (s, e), xytext=(5, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("incertidumbre σ"); ax.set_ylabel("E[f] del óptimo"); ax.set_ylim(0, 1.1); ax.legend()
    ax.set_title("El costo de la robustez crece con la incertidumbre"); fig.tight_layout(); plt.show()


def mapas_ensamble(logK, n=10, vmin=-8.5, vmax=-4.5, columnas=5, titulo=None, nombre="geología"):
    """Las primeras n geologías de un ensamble (campos de ln K) con los pozos y el humedal, en una cuadrícula."""
    import matplotlib.pyplot as plt
    n = min(n, len(logK)); filas = int(np.ceil(n / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(4 * columnas, 3.8 * filas))
    for k, ax in enumerate(np.ravel(ejes)):
        if k >= n:
            ax.axis("off"); continue
        im = ax.imshow(logK[k], origin="lower", extent=[0, LADO, 0, LADO], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        for j, (x, y) in enumerate(pozos_xy()):
            ax.plot(x, y, "o", ms=7, mfc="yellow", mec="k"); ax.annotate(str(j + 1), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=9)
        ax.plot(*humedal_xy(), "P", ms=9, mfc="#8fd18f", mec="k")
        ax.set_title(f"{nombre} {k}"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ejes, label="ln K", shrink=0.6)
    if titulo: fig.suptitle(titulo, fontsize=14)
    plt.show()


def resumen_ensamble(logK, vmin=-8.5, vmax=-4.5):
    """Lo que solo se ve juntando todas las geologías: el promedio de ln K y su desviación estándar celda por celda, con los pozos y el humedal.
    Devuelve (media, desviación) en cada pozo."""
    import matplotlib.pyplot as plt
    L = np.asarray(logK); m = L.mean(axis=0); s = L.std(axis=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
    i1 = a1.imshow(m, origin="lower", extent=[0, LADO, 0, LADO], cmap="RdBu_r", vmin=vmin, vmax=vmax)
    i2 = a2.imshow(s, origin="lower", extent=[0, LADO, 0, LADO], cmap="viridis", vmin=0, vmax=max(1.2, float(s.max())))
    en_pozos = []
    for ax in (a1, a2):
        for j, (x, y) in enumerate(pozos_xy()):
            ax.plot(x, y, "o", ms=9, mfc="yellow", mec="k"); ax.annotate(f"pozo {j + 1}", (x, y), xytext=(6, 5), textcoords="offset points", fontsize=10, color="k" if ax is a1 else "w")
        ax.plot(*humedal_xy(), "P", ms=11, mfc="#8fd18f", mec="k"); ax.set_xticks([]); ax.set_yticks([])
    for x, y in pozos_xy():
        i, k = min(int(y / LADO * L.shape[1]), L.shape[1] - 1), min(int(x / LADO * L.shape[2]), L.shape[2] - 1)
        en_pozos.append((float(m[i, k]), float(s[i, k])))
    a1.set_title(f"El promedio de ln K en las {len(L)} geologías"); a2.set_title("La desviación estándar de ln K")
    fig.colorbar(i1, ax=a1, label="ln K promedio", shrink=0.8); fig.colorbar(i2, ax=a2, label="desviación de ln K", shrink=0.8)
    fig.tight_layout(); plt.show()
    return en_pozos


def lnK_en_pozos(logK):
    """Cajas con el ln K de cada pozo en todas las geologías del ensamble: cuánto varía la conductividad donde está cada pozo.
    Devuelve una lista (mediana, percentil 5, percentil 95) por pozo."""
    import matplotlib.pyplot as plt
    L = np.asarray(logK); datos = []
    for x, y in pozos_xy():
        i, k = min(int(y / LADO * L.shape[1]), L.shape[1] - 1), min(int(x / LADO * L.shape[2]), L.shape[2] - 1)
        datos.append(L[:, i, k])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.boxplot(datos, widths=0.5, patch_artist=True, boxprops=dict(facecolor="#c9d6e4"), medianprops=dict(color=_ROJO, lw=2))
    for j, d in enumerate(datos):
        ax.plot(np.random.default_rng(j).normal(j + 1, 0.06, len(d)), d, "o", ms=3, color=_AZUL, alpha=0.35)
    ax.set_xticks(range(1, len(datos) + 1)); ax.set_xticklabels([f"pozo {j + 1}" for j in range(len(datos))])
    ax.set_ylabel("ln K en el pozo"); ax.set_title(f"La conductividad donde está cada pozo, en las {len(L)} geologías")
    fig.tight_layout(); plt.show()
    return [(float(np.median(d)), float(np.percentile(d, 5)), float(np.percentile(d, 95))) for d in datos]


def decision_en_ensamble(S, Hh, s_hom, h_hom, n=8, s_max=3.0, s_hum=1.25):
    """Una decisión en el ensamble, en dos paneles. Izquierda: barras de los tres pozos y el humedal en las n primeras geologías (coral si se pasa del límite),
    con el límite y lo que prometía el homogéneo. Derecha: pozo 1 contra pozo 2 en todas las geologías, con el cuadro donde los dos cumplen; el color dice si la
    decisión rompe alguna regla (contando el pozo 3 y el humedal). S: (N, 3); Hh: (N,)."""
    import matplotlib.pyplot as plt
    S = np.asarray(S); Hh = np.asarray(Hh); N = len(S)
    vals = np.column_stack([S, Hh]); lim = np.array([s_max] * 3 + [s_hum]); hom = np.append(s_hom, h_hom)
    rompe = (S > s_max + 1e-9).any(axis=1) | (Hh > s_hum + 1e-9)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    nombres = ("pozo 1", "pozo 2", "pozo 3", "humedal"); w = 0.2
    for k in range(n):
        for j in range(4):
            v = vals[k, j]; x = k + (j - 1.5) * w
            a1.bar(x, min(v, 12), w * 0.9, color=_ROJO if v > lim[j] + 1e-9 else _VERDE, edgecolor="k", lw=0.4)
            a1.plot([x - w * 0.45, x + w * 0.45], [hom[j], hom[j]], "-", color=_ORO, lw=3.2, label="lo que prometía el homogéneo" if k == j == 0 else None)
            a1.plot([x - w * 0.45, x + w * 0.45], [lim[j], lim[j]], "--", color="k", lw=1.4, label="límite (3 m en los pozos, 1.25 m en el humedal)" if k == j == 0 else None)
    a1.set_xticks(range(n)); a1.set_xticklabels([f"geología {k}\n" + ("ROMPE" if rompe[k] else "cumple") for k in range(n)], fontsize=9)
    a1.set_ylabel("abatimiento [m]"); a1.set_ylim(0, 12.5); a1.legend(fontsize=9, loc="upper left")
    a1.set_title("Las primeras geologías: barras de izquierda a derecha = pozo 1, 2, 3 y humedal (coral: se pasa)", fontsize=10.5)
    a2.add_patch(plt.Rectangle((0, 0), s_max, s_max, color=_VERDE, alpha=0.15, label="los dos pozos cumplen"))
    a2.scatter(S[~rompe, 0], S[~rompe, 1], s=22, color=_VERDE, edgecolor="k", lw=0.3, label=f"cumple todo ({(~rompe).sum()})")
    a2.scatter(S[rompe, 0], np.minimum(S[rompe, 1], 24.5), s=22, color=_ROJO, edgecolor="k", lw=0.3, alpha=0.8, label=f"rompe alguna regla ({rompe.sum()})")
    a2.plot(s_hom[0], s_hom[1], "*", ms=20, color=_ORO, mec="k", label="lo que prometía el homogéneo")
    a2.axvline(s_max, color=_ROJO, ls="--", lw=1.2); a2.axhline(s_max, color=_ROJO, ls="--", lw=1.2)
    a2.set_xlim(0, max(16, float(S[:, 0].max()) + 0.5)); a2.set_ylim(0, 25)
    a2.set_xlabel("abatimiento del pozo 1 [m]"); a2.set_ylabel("abatimiento del pozo 2 [m]"); a2.legend(fontsize=9, loc="upper right")
    a2.set_title(f"Las {N} geologías: cada punto es una", fontsize=11)
    fig.tight_layout(); plt.show()


def tres_promedios(X, limite=3.0, unidad="m"):
    """La misma operación, el promedio, con tres h distintas (2026-11-23, ARP: «una visualización de esta celda»). Tres paneles, una barra por geología:
    h(X) = X (su promedio es la media), h(X) = (X − media)² (la varianza) y h(X) = 1 si X > límite (la probabilidad). La raya horizontal es el promedio de
    las barras de cada panel."""
    import matplotlib.pyplot as plt
    X = np.asarray(X, float); n = len(X); k = np.arange(n); m = X.mean(); v = (X - m) ** 2; ind = (X > limite).astype(float)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    a1, a2, a3 = axs
    a1.bar(k, X, 1.0, color=_AZUL, alpha=0.75); a1.axhline(m, color=_ORO, lw=2.5)
    caja = dict(fc="white", ec=_ORO, alpha=0.92, boxstyle="round,pad=0.3")
    a1.text(0.97, 0.95, f"media = {m:.2f} {unidad}", transform=a1.transAxes, color=_ORO, fontsize=12, fontweight="bold", ha="right", va="top", bbox=caja)
    a1.set_title("h(X) = X  →  la media", fontsize=13); a1.set_ylabel(f"h(X) [{unidad}]")
    a2.bar(k, v, 1.0, color=_GRIS, alpha=0.75); a2.axhline(v.mean(), color=_ORO, lw=2.5)
    a2.text(0.97, 0.95, f"varianza = {v.mean():.2f} {unidad}²\n(σ = {np.sqrt(v.mean()):.2f} {unidad})", transform=a2.transAxes, color=_ORO, fontsize=12, fontweight="bold", ha="right", va="top", bbox=caja)
    a2.set_title("h(X) = (X − media)²  →  la varianza", fontsize=13); a2.set_ylabel(f"h(X) [{unidad}²]")
    a3.bar(k, np.where(ind > 0, 1.0, 0.06), 1.0, color=[_ROJO if i else _VERDE for i in ind], alpha=0.85); a3.axhline(ind.mean(), color=_ORO, lw=2.5)
    a3.text(0.97, 0.97, f"P(X > {limite:g}) = {ind.mean():.2f}", transform=a3.transAxes, color=_ORO, fontsize=12, fontweight="bold", ha="right", va="top", bbox=caja)
    a3.text(0.02, 0.97, "rojo = 1 (pasa)   verde = 0 (no pasa)", transform=a3.transAxes, fontsize=10, color=_GRIS, va="top")
    a3.set_title(f"h(X) = 1 si X > {limite:g} {unidad}, 0 si no  →  la probabilidad", fontsize=13); a3.set_ylim(0, 1.35); a3.set_yticks([0, 1])
    for ax in axs:
        ax.set_xlabel("geología (una barra por geología)"); ax.set_xlim(-1, n)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.suptitle("La misma operación en los tres: el promedio de las barras (la raya dorada). Solo cambia h.", fontsize=13, y=1.02)
    fig.tight_layout(); plt.show()


def resumen_variable(X, limite=3.0, prometido=3.0, unidad="m"):
    """La celda de media, varianza y P(X > límite) en una sola figura (2026-11-23, ARP: «como el histograma del video de la lámina 12»): el histograma de X
    (una barra cada medio metro), la media con su ± σ, y las barras que pasan del límite en otro color con la probabilidad escrita."""
    import matplotlib.pyplot as plt
    X = np.asarray(X, float); m = X.mean(); sd = X.std(); p = np.mean(X > limite)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bins = np.arange(0, np.ceil(X.max()) + 0.5, 0.5)
    n_, b_, parches = ax.hist(X, bins=bins, color="#5B9BFF", edgecolor="white")
    for pa, izq in zip(parches, b_[:-1]):
        if izq >= limite: pa.set_facecolor("#E8A09A")
    top = max(n_) * 1.18; ax.set_ylim(0, top)
    ax.annotate("", (m - sd, top * 0.74), (m + sd, top * 0.74), arrowprops=dict(arrowstyle="<->", color=_ORO, lw=2))
    ax.text(m + sd + 0.2, top * 0.74, f"media ± σ  (σ = {sd:.2f} {unidad}; varianza {sd ** 2:.2f} {unidad}²)", color=_ORO, va="center", fontsize=11, fontweight="bold")
    ax.axvline(m, color=_ROJO, lw=2.8); ax.text(m + 0.15, top * 0.93, f"media = {m:.2f} {unidad}", color=_ROJO, fontsize=13, fontweight="bold")
    ax.axvline(limite, color="k", ls="--", lw=1.6)
    ax.text(0.0, top * 0.97, f"punteada: {prometido:.2f} {unidad}\n(un solo modelo\ny el límite)", color="k", fontsize=10, ha="left", va="top")
    ax.text(X.max(), top * 0.45, f"en rojo, las que pasan de {limite:g} {unidad}:\nP(X > {limite:g}) = {p:.2f}", color=_ROJO, fontsize=12, fontweight="bold", ha="right")
    ax.set_xlabel(f"abatimiento del pozo 1 [{unidad}] (una barra por cada medio metro)"); ax.set_ylabel("geologías")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); plt.show()


def variable_aleatoria(X, nombre="abatimiento del pozo 1 [m]", limite=None, semilla=0):
    """Introducción básica a una variable aleatoria, en dos paneles. Izquierda: un dado lanzado 12, 120 y 12 000 veces; la frecuencia de cada cara se acerca a 1/6
    (su distribución). Derecha: X, una variable aleatoria del acuífero (un valor por geología): su histograma es una estimación de su distribución p(x)."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(semilla); X = np.asarray(X)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6))
    for k, (n, c) in enumerate(((12, _GRIS), (120, _AZUL), (12000, _ORO))):
        caras = rng.integers(1, 7, n); f = np.bincount(caras, minlength=7)[1:] / n
        a1.bar(np.arange(1, 7) + (k - 1) * 0.27, f, 0.25, color=c, label=f"{n} lanzamientos")
    a1.axhline(1 / 6, color="k", ls="--", lw=1.5, label="1/6: la distribución del dado")
    a1.set_xticks(range(1, 7)); a1.set_xlabel("cara del dado (el valor x que toma X)"); a1.set_ylabel("fracción de lanzamientos")
    a1.set_title("Un dado: no sé qué saldrá, pero sé con qué probabilidad"); a1.legend(fontsize=9)
    a2.hist(X, bins=30, density=True, color="#2E7DB5", edgecolor="k", alpha=0.85, label=f"{len(X)} valores, uno por geología")
    if limite is not None:
        a2.axvline(limite, color=_ROJO, ls="--", lw=2, label=f"límite: {limite:g} m")
    a2.set_xlabel(nombre); a2.set_ylabel("p(x) estimada"); a2.legend(fontsize=9)
    a2.set_title("El acuífero: no sé qué geología es la real, pero tengo 200 posibles")
    fig.tight_layout(); plt.show()


def tres_objetivos(p, s, costo, indice, titulo="Cómo se arma cada objetivo"):
    """Los tres objetivos de una decisión, pozo por pozo: abatimiento s_i, costo c_i p_i (H_i − h_i⁰ + s_i) y término del índice exp(−k (c_i⁰ + a_i p_i)).
    Cada panel suma sus tres barras: ese es el objetivo."""
    import matplotlib.pyplot as plt
    pozos = [f"pozo {j + 1}\n{p[j]:.0f} m³/d" for j in range(3)]
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, v, nom, u, c in zip(ejes, (s, costo, indice), ("Abatimiento total", "Costo", "Índice de necesidad"), ("m", "$/d", ""), (_AZUL, _ORO, _VERDE)):
        v = np.asarray(v, dtype=float); b = ax.bar(pozos, v, color=c, edgecolor="k")
        for r, x in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, x, f"{x:.2f}" if x < 10 else f"{x:.1f}", ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{nom} = {v.sum():.2f} {u}".strip(), fontsize=12); ax.set_ylim(0, v.max() * 1.25)
    ejes[0].set_ylim(0, max(3.4, float(np.max(s)) * 1.25)); ejes[0].axhline(3, color=_ROJO, ls="--", lw=1.5); ejes[0].text(2.45, 3.07, "límite 3 m por pozo", color=_ROJO, ha="right")
    fig.suptitle(titulo, fontsize=13); fig.tight_layout(); plt.show()


def decision_en_matrices(casos, s_max=3.0, s_hum=1.25):
    """La misma decisión con varias matrices (el homogéneo y algunas geologías): barras de los tres pozos y el humedal por caso, coral si se pasa.
    casos: lista de (nombre, s3, sh, abatimiento_total, costo)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 4.6)); w = 0.2; lim = [s_max] * 3 + [s_hum]
    for k, (nombre, s3, sh, S, C) in enumerate(casos):
        vals = list(s3) + [sh]
        for j, v in enumerate(vals):
            x = k + (j - 1.5) * w
            ax.bar(x, v, w * 0.9, color=_ROJO if v > lim[j] + 1e-9 else _VERDE, edgecolor="k", lw=0.4)
            ax.plot([x - w * 0.45, x + w * 0.45], [lim[j], lim[j]], "--", color="k", lw=1.4, label="límite (3 m en los pozos, 1.25 m en el humedal)" if k == j == 0 else None)
            ax.text(x, v + 0.12, f"{v:.1f}", ha="center", fontsize=8.5)
        ax.text(k, -1.25, f"total {S:.1f} m · {C:.0f} $/d", ha="center", fontsize=10, color=_AZUL)
    ax.set_xticks(range(len(casos))); ax.set_xticklabels([c[0] for c in casos], fontsize=11)
    ax.set_ylim(-1.6, max(max(list(c[1]) + [c[2]]) for c in casos) * 1.15); ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("abatimiento [m]"); ax.legend(fontsize=9, loc="upper right")
    ax.set_title("La misma decisión con cuatro matrices: barras de izquierda a derecha = pozo 1, 2, 3 y humedal (coral: se pasa)", fontsize=11)
    fig.tight_layout(); plt.show()


def bombeos_frente(X, indice, titulo="Qué decisiones hay en el frente", S=None, Sh=None, s_max=3.0, s_hum=1.25,
                   titulo_abajo="Cuánto baja cada pozo y el humedal con esas decisiones (modelo homogéneo)"):
    """Las decisiones de un frente ordenadas por el índice de necesidad: cuánto bombea cada pozo (barras apiladas) y el índice (línea).
    A la izquierda, poco bombeo e índice alto; a la derecha, mucho bombeo e índice bajo. Si se dan S (n x 3, abatimiento de cada pozo) y Sh (n, el del
    humedal), un segundo panel debajo, con las decisiones en el mismo orden, dibuja cuánto baja cada pozo y el humedal, con sus límites (2026-11-23, ARP)."""
    import matplotlib.pyplot as plt
    X = np.asarray(X); indice = np.asarray(indice); o = np.argsort(-indice); Xo = X[o]
    if S is None:
        fig, ax = plt.subplots(figsize=(13, 4.4))
    else:
        fig, (ax, ab) = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
    x = np.arange(len(Xo)); base = np.zeros(len(Xo))
    for j, c in enumerate((_AZUL, _ORO, _VERDE)):
        ax.bar(x, Xo[:, j], 1.0, bottom=base, color=c, label=f"pozo {j + 1}"); base += Xo[:, j]
    ax.set_ylabel("bombeo [m³/d]")
    a2 = ax.twinx(); a2.plot(x, indice[o], color="k", lw=2, label="índice de necesidad"); a2.set_ylabel("índice de necesidad")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=9, ncol=4)
    ax.set_title(titulo, fontsize=12)
    eje_x = f"las {len(Xo)} decisiones, de la que menos bombea a la que más (ordenadas por el índice)"
    if S is None:
        ax.set_xlabel(eje_x)
    else:
        S = np.asarray(S)[o]
        for j, c in enumerate((_AZUL, _ORO, _VERDE)):
            ab.plot(x, S[:, j], color=c, lw=1.6, label=f"pozo {j + 1}")
        ab.axhline(s_max, color=_ROJO, ls="--", lw=2); ab.text(len(Xo) * 0.995, s_max + 0.05, f"límite de los pozos: {s_max:g} m", color=_ROJO, fontsize=10, va="bottom", ha="right")
        if Sh is not None:
            ab.plot(x, np.asarray(Sh)[o], color="k", lw=1.6, ls=":", label="humedal")
            ab.axhline(s_hum, color="k", ls="--", lw=1.2); ab.text(len(Xo) * 0.005, s_hum + 0.05, f"límite del humedal: {s_hum:g} m", color="k", fontsize=10, va="bottom")
        ab.set_ylim(0, max(s_max, float(S.max())) * 1.12)
        ab.set_ylabel("cuánto baja [m]"); ab.set_xlabel(eje_x); ab.legend(loc="upper left", fontsize=9, ncol=4)
        ab.set_title(titulo_abajo, fontsize=11)
    fig.tight_layout(); plt.show()


def receta_propagacion(logK, A, H, p, n_mapas=3, s_max=3.0, s_hum=1.25):
    """La receta de propagación en cuatro paneles, con los datos del ensamble: 1 caracterizar (la distribución de ln K), 2 generar (algunas geologías),
    3 simular (la decisión p en esas geologías: s = α_k p) y 4 hacer estadística (el pozo 1 en todas, con su media)."""
    import matplotlib.pyplot as plt
    L = np.asarray(logK); A = np.asarray(A); H = np.asarray(H); p = np.asarray(p, dtype=float)
    S = A @ p; Hh = H @ p
    fig = plt.figure(figsize=(16, 4.6)); gs = fig.add_gridspec(2, 3 + n_mapas, height_ratios=[1, 1])
    a1 = fig.add_subplot(gs[:, 0]); a1.hist(L.ravel(), bins=50, color=_GRIS, density=True)
    a1.set_title("1 · Caracterizar\nla distribución de ln K", fontsize=11); a1.set_xlabel("ln K"); a1.set_yticks([])
    for k in range(n_mapas):
        ax = fig.add_subplot(gs[0, 1 + k]); ax.imshow(L[k], origin="lower", extent=[0, LADO, 0, LADO], cmap="RdBu_r", vmin=-8.5, vmax=-4.5)
        for x, y in pozos_xy(): ax.plot(x, y, "o", ms=4, mfc="yellow", mec="k")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"geología {k}", fontsize=9)
        if k == n_mapas // 2: ax.text(0.5, 1.32, "2 · Generar realizaciones", transform=ax.transAxes, ha="center", fontsize=11)
        bx = fig.add_subplot(gs[1, 1 + k]); vals = list(S[k]) + [Hh[k]]; lim = [s_max] * 3 + [s_hum]
        bx.bar(range(4), vals, color=[_ROJO if v > l + 1e-9 else _VERDE for v, l in zip(vals, lim)], edgecolor="k", lw=0.4)
        for j, l in enumerate(lim): bx.plot([j - 0.4, j + 0.4], [l, l], "--", color="k", lw=1)
        bx.set_xticks(range(4)); bx.set_xticklabels(["p1", "p2", "p3", "hum"], fontsize=8); bx.tick_params(axis="y", labelsize=8)
        if k == 0: bx.set_ylabel("s [m]", fontsize=9)
        if k == n_mapas // 2: bx.set_title("3 · Simular cada una: s = α·p", fontsize=11)
    a4 = fig.add_subplot(gs[:, 1 + n_mapas:]); a4.hist(S[:, 0], bins=30, color="#2E7DB5", edgecolor="k")
    a4.axvline(S[:, 0].mean(), color=_ORO, lw=2.5, label=f"media {S[:, 0].mean():.2f} m"); a4.axvline(s_max, color=_ROJO, ls="--", lw=2, label=f"límite {s_max:g} m")
    a4.set_title(f"4 · Hacer estadística\nel pozo 1 en las {len(S)} geologías", fontsize=11); a4.set_xlabel("abatimiento del pozo 1 [m]"); a4.legend(fontsize=9)
    fig.tight_layout(); plt.show()


def funcion_efectiva(F_geo, F_hom, n_marcas=5, ejes=("Abatimiento total [m]", "Costo [$/d]")):
    """Cómo se forma la función efectiva de una decisión: para cada objetivo, su valor en cada geología (puntos; las primeras n_marcas numeradas)
    y el promedio con las primeras N geologías (línea), que termina en f_eff; la línea punteada es lo que prometía un solo modelo."""
    import matplotlib.pyplot as plt
    F_geo = np.asarray(F_geo); N = np.arange(1, len(F_geo) + 1)
    fig, ejes_ = plt.subplots(1, 2, figsize=(15, 4.5))
    for j, (ax, nom) in enumerate(zip(ejes_, ejes)):
        v = F_geo[:, j]; prom = np.cumsum(v) / N
        ax.scatter(N, v, s=12, color="#2E9BB5", alpha=0.5, label="f(y): la decisión en la geología k")
        ax.scatter(N[:n_marcas], v[:n_marcas], s=60, color=_ORO, edgecolor="k", zorder=4, label=f"las geologías 0 a {n_marcas - 1} de la tabla")
        ax.plot(N, prom, color=_VERDE, lw=2.8, label=f"promedio con las primeras N: termina en f_eff = {prom[-1]:.1f}")
        ax.axhline(F_hom[j], color=_ORO, ls="--", lw=2, label=f"lo que prometía un solo modelo: {F_hom[j]:.1f}")
        ax.set_xlabel("geología k (N = cuántas llevamos)"); ax.set_ylabel(nom.replace("$", r"\$")); ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"{nom.split(' [')[0]}: sumar los {len(v)} resultados y dividir entre {len(v)}", fontsize=11)
    fig.tight_layout(); plt.show()


def suma_acumulada(F_geo, ejes=("Abatimiento total [m]", "Costo [$/d]")):
    """La integral como suma: para cada objetivo, la suma de los resultados de la decisión con las primeras N geologías (crece con cada geología)."""
    import matplotlib.pyplot as plt
    F_geo = np.asarray(F_geo); N = np.arange(1, len(F_geo) + 1)
    fig, ejes_ = plt.subplots(1, 2, figsize=(15, 4.2))
    for j, (ax, nom) in enumerate(zip(ejes_, ejes)):
        c = np.cumsum(F_geo[:, j]); ax.fill_between(N, 0, c, color="#2E9BB5", alpha=0.25); ax.plot(N, c, color=_AZUL, lw=2.5)
        ax.plot(N[-1], c[-1], "o", ms=10, color=_ORO, mec="k")
        ax.text(0.03, 0.9, f"suma de los {len(c)}: {c[-1]:,.0f}".replace(",", " "), transform=ax.transAxes, fontsize=12, color=_AZUL, fontweight="bold")
        ax.set_xlabel("geologías sumadas (N)"); ax.set_ylabel(("suma del " + nom.lower()).replace("$", r"\$")); ax.set_title(f"{nom.split(' [')[0]}: cada geología agrega su resultado", fontsize=11)
    fig.tight_layout(); plt.show()


def cuenta_evaluaciones(llamadas, n_geologias, titulo="Cuántas veces se evalúa el modelo"):
    """Lo que cuesta la robustez en cómputo: llamadas a la función objetivo (iguales en los dos casos) y simulaciones del modelo (×N geologías en el robusto).
    llamadas: número de llamadas de una corrida."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2))
    et = ["llamadas a la función objetivo\n(determinista y robusto)", "simulaciones del modelo\n(determinista: 1 por llamada)", f"simulaciones del modelo\n(robusto: {n_geologias} por llamada)"]
    v = [llamadas, llamadas, llamadas * n_geologias]; b = ax.barh(et, v, color=[_GRIS, _AZUL, _ROJO])
    for r, x in zip(b, v): ax.text(x * 1.15, r.get_y() + r.get_height() / 2, f"{x:,}".replace(",", " "), va="center", fontsize=12, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(1, v[-1] * 20); ax.set_xlabel("número (escala logarítmica)"); ax.invert_yaxis(); ax.set_title(titulo, fontsize=12)
    fig.tight_layout(); plt.show()


def enjambre_dos_calificaciones(F_det, F_rob, nd_det, nd_rob, ejes=("Abatimiento total [m]", "Costo [$/d]")):
    """El mismo enjambre calificado de dos maneras: con un solo modelo (izquierda) y con el promedio de las geologías (derecha). En oro, las partículas que no
    domina ninguna otra: las que entrarían al archivo. Las flechas grises unen cada partícula con su calificación robusta."""
    import matplotlib.pyplot as plt
    F_det = np.asarray(F_det); F_rob = np.asarray(F_rob)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, F, nd, tit in ((a1, F_det, nd_det, "calificado con un solo modelo"), (a2, F_rob, nd_rob, "calificado con el promedio de las geologías")):
        ax.scatter(F[:, 0], F[:, 1], s=40, color="#2E9BB5", edgecolor="k", lw=0.3, label="partícula")
        ax.scatter(F[nd, 0], F[nd, 1], s=110, color=_ORO, edgecolor="k", zorder=4, label=f"no dominada: entra al archivo ({int(np.sum(nd))})")
        ax.set_xlabel(ejes[0]); ax.set_title(tit, fontsize=12); ax.legend(fontsize=9, loc="upper left")
    for (x0, y0) in F_det[:0]: pass
    a1.set_ylabel(ejes[1].replace("$", r"\$")); a1.set_xlim(0, max(F_rob[:, 0].max(), F_det[:, 0].max()) * 1.05); a2.set_xlim(a1.get_xlim())
    fig.suptitle("El mismo enjambre, dos calificaciones: OMOPSO solo ve los números que le devuelve la función", fontsize=13); fig.tight_layout(); plt.show()


def castigo_en_promedio(s_prom, h_prom, castigo, s_max=3.0, s_hum=1.25):
    """El castigo del OMOPSO robusto: para cada partícula, el abatimiento PROMEDIO de su pozo más exigido (barras, coral si pasa de 3 m) y el castigo M·v que
    se suma a sus objetivos (puntos)."""
    import matplotlib.pyplot as plt
    s_prom = np.asarray(s_prom); peor = s_prom.max(axis=1); o = np.argsort(peor); x = np.arange(len(peor))
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.bar(x, peor[o], color=[_ROJO if v > s_max + 1e-9 else _VERDE for v in peor[o]], edgecolor="k", lw=0.3)
    ax.axhline(s_max, color="k", ls="--", lw=1.5, label="límite: 3 m en promedio")
    ax.set_xlabel("las partículas del enjambre, de la que menos abate a la que más"); ax.set_ylabel("abatimiento promedio del pozo más exigido [m]")
    a2 = ax.twinx(); a2.plot(x, np.asarray(castigo)[o], "o", color=_ORO, mec="k", label="castigo M·v que se suma a los objetivos"); a2.set_ylabel("castigo")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.set_title(f"Las reglas se revisan con el promedio: {int((peor > s_max + 1e-9).sum())} de {len(peor)} partículas pagan castigo por algún pozo", fontsize=12)
    fig.tight_layout(); plt.show()


def llamadas_contra_tiempo(escenarios, n_llamadas, n_geologias=200, referencias=()):
    """Tiempo de cómputo contra número de llamadas a la función objetivo (log-log). escenarios: lista de (segundos por llamada, nombre, de dónde sale, color,
    estilo). Marca la corrida de n_llamadas con su duración y escribe junto a cada recta de dónde sale el tiempo por corrida."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    n = np.logspace(0, np.log10(n_llamadas * 4), 200)
    fig, ax = plt.subplots(figsize=(13, 7.2))
    Ts = sorted(n_llamadas * e[0] for e in escenarios)
    for k, (t1, nombre, fuente, c, ls) in enumerate(escenarios):
        ax.plot(n, n * t1, ls, color=c, lw=2.4, label=f"{nombre}: {fuente}")
        T = n_llamadas * t1; ax.plot(n_llamadas, T, "o", ms=9, color=c, mec="k", zorder=5)
        j = Ts.index(T)                                     # separa los rótulos que caen juntos: el de abajo baja, el de arriba sube un poco
        dy = -16 if (j + 1 < len(Ts) and Ts[j + 1] / T < 3) else (2 if (j > 0 and T / Ts[j - 1] < 3) else -4)
        ax.annotate(_duracion(T), (n_llamadas, T), xytext=(9, dy), textcoords="offset points", fontsize=11, color=c, fontweight="bold")
        xl = 1.15 * 6 ** (k % 3)                            # los nombres, escalonados a lo largo del eje
        ax.text(xl, xl * t1 * 2.2, nombre, color=c, fontsize=9.5, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_xscale("log"); ax.set_yscale("log")
    for seg, et, c in referencias:                          # tiempos totales publicados (una línea horizontal con su origen)
        ax.axhline(seg, color=c, lw=2, ls=(0, (6, 2, 1, 2))); ax.text(1.15, seg * 1.25, et, color=c, fontsize=9.5, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    for seg, et in ((1, "1 segundo"), (60, "1 minuto"), (3600, "1 hora"), (86400, "1 día"), (86400 * 365, "1 año")):
        ax.axhline(seg, color=_GRIS, lw=0.8, ls=":"); ax.text(n[-1], seg * 1.12, et, color=_GRIS, fontsize=9, ha="right", va="bottom")
    ax.axvline(n_llamadas, color=_ORO, lw=1.5)
    ax.text(n_llamadas / 1.08, ax.get_ylim()[0] * 2, f"la corrida de la 2.3:\n{n_llamadas:,} llamadas".replace(",", " "), color=_ORO, fontsize=10, ha="right", va="bottom")
    def etiqueta(v, _):
        e = int(round(np.log10(v))) if v > 0 else 0
        return f"$10^{{{e}}}$\n({v:,.0f})".replace(",", " ") if abs(np.log10(v) - e) < 1e-6 else ""
    ax.xaxis.set_major_formatter(FuncFormatter(etiqueta))
    ax.set_xlabel("número de llamadas a la función objetivo"); ax.set_ylabel("tiempo de cómputo [s]")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)
    ax.set_title(f"El mismo número de llamadas; en el robusto, {n_geologias} simulaciones en cada una", fontsize=12)
    fig.tight_layout(); plt.show()

def _duracion(seg):
    """Un tiempo en segundos, en la unidad más legible."""
    for u, d in (("años", 86400 * 365), ("días", 86400), ("h", 3600), ("min", 60), ("s", 1)):
        if seg >= d: return f"{seg / d:.1f} {u}"
    return f"{seg * 1e3:.1f} ms"


def histograma_riesgo(riesgos, riesgo_max=None, titulo="¿Qué tan seguido falla cada frente?", xl="riesgo en 200 geologías de validación"):
    """Histogramas superpuestos del riesgo de las decisiones de cada frente. riesgos: {nombre: (valores, color)}; riesgo_max dibuja una línea."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    for nombre, (r, color) in riesgos.items():
        ax.hist(np.asarray(r), bins=np.linspace(0, 1, 21), color=color, alpha=0.6, label=f"{nombre} (media {np.mean(r):.0%})", edgecolor="k")
    if riesgo_max is not None:
        ax.axvline(riesgo_max, color="#1f5fbf", ls="--", lw=2, label=f"riesgo pedido: {riesgo_max:.0%}")
    ax.set_xlabel(xl); ax.set_ylabel("decisiones"); ax.legend(fontsize=9); ax.set_title(titulo); fig.tight_layout(); plt.show()


def promedio_contra_riesgo(frentes, marca=None, s_max=3.0, ejemplos=()):
    """Cumplir en promedio no es cumplir: cada decisión como un punto, con el abatimiento promedio del pozo más exigido (eje x) y su riesgo (eje y).
    frentes: {nombre: (promedios, riesgos, color)}; marca: (promedio, riesgo, texto) de una decisión a señalar;
    ejemplos: [(promedio, riesgo, texto, color, (dx, dy))] puntos de ejemplo con un círculo y su rótulo (dx, dy: corrimiento del rótulo, en m y %)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    xmax = max(float(np.max(v[0])) for v in frentes.values()) * 1.05 + 0.3
    ax.axvspan(0, s_max, color=_VERDE, alpha=0.12, label=f"cumple en promedio (abajo de {s_max:g} m)")
    ax.axvline(s_max, color=_ROJO, ls="--", lw=1.5)
    for nombre, (prom, r, color) in frentes.items():
        ax.scatter(prom, 100 * np.asarray(r), s=22, color=color, edgecolor="k", lw=0.3, alpha=0.75, label=f"{nombre} ({len(prom)} decisiones)")
    if marca is not None:
        ax.plot(marca[0], 100 * marca[1], "*", ms=22, color=_ORO, mec="k", zorder=5)
        ax.annotate(f"{marca[2]}:\ncumple en promedio\ny rompe en {marca[1]:.0%}", (marca[0], 100 * marca[1]), xytext=(marca[0] + 5.0, 100 * marca[1] - 30),
                    fontsize=10.5, color=_ORO, fontweight="bold", bbox=dict(fc="white", ec=_ORO), arrowprops=dict(arrowstyle="->", color=_ORO, lw=1.5))
    for x, r, texto, color, (dx, dy) in ejemplos:
        ax.plot(x, 100 * r, "o", ms=16, mfc="none", mec=color, mew=2.5, zorder=6)
        ax.annotate(texto, (x, 100 * r), xytext=(x + dx, 100 * r + dy), fontsize=9.5, color=color, fontweight="bold",
                    bbox=dict(fc="white", ec=color), arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=7)
    ax.set_xlim(0, xmax); ax.set_ylim(-3, 103)
    ax.set_xlabel("abatimiento promedio del pozo más exigido [m]"); ax.set_ylabel("riesgo: geologías en que rompe [%]")
    ax.legend(fontsize=9, loc="lower right"); ax.set_title("El promedio contra el riesgo: cada punto es una decisión")
    fig.tight_layout(); plt.show()


def rio_promedio(s_geo, falla, s_max=3.0):
    """Cumplir en promedio no es cumplir, la misma figura de la lámina 22 de la S4 (laminas/figuras_conceptuales.py, ARP 2026-11-23).
    Izquierda, el río de un metro de profundidad media con un canal de 3 m. Derecha, una decisión geología por geología.
    s_geo: (N,) abatimiento del pozo más exigido en cada geología [m]; falla: (N,) True donde la decisión rompe alguna regla."""
    import matplotlib.pyplot as plt
    AZ, VE, CO, TI = "#0B5B72", "#2E9B4F", "#D9534F", "#1B2B33"
    with plt.rc_context({"font.size": 13, "axes.edgecolor": TI, "axes.labelcolor": TI, "xtick.color": TI, "ytick.color": TI}):   # la letra de la lámina
        s_geo = np.asarray(s_geo, dtype=float); falla = np.asarray(falla, dtype=bool)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.0, 1.05]))
        x = np.linspace(0, 20, 400)
        prof = 0.55 + 1.85 * np.exp(-((x - 12.5) / 1.6) ** 2); prof = prof * (1.0 / prof.mean())
        a1.fill_between(x, 0, -prof, color="#9CC9DB"); a1.plot(x, -prof, color="#6B4F2A", lw=3); a1.axhline(0, color=AZ, lw=2)
        a1.axhline(-prof.mean(), color=VE, ls="--", lw=2.5)
        a1.text(0.3, -prof.mean() - 0.45, f"en promedio: {prof.mean():.0f} m  ✓", color=VE, fontsize=14, fontweight="bold")
        a1.text(12.5, -prof.max() - 0.35, f"aquí: {prof.max():.0f} m  ✗", ha="center", color=CO, fontsize=14, fontweight="bold")
        for xp in (3.0, 12.5):                                    # una persona de 1.6 m
            y0 = -np.interp(xp, x, prof)
            a1.plot([xp, xp], [y0, y0 + 1.25], color=TI, lw=3); a1.plot(xp, y0 + 1.45, "o", ms=11, color=TI)
        a1.set_ylim(-3.4, 2.0); a1.set_xticks([]); a1.set_yticks([])
        for sp in a1.spines.values(): sp.set_visible(False)
        a1.set_title("Un río: «en promedio tiene un metro»", color=TI, fontsize=15, fontweight="bold")
        bins = np.arange(0, min(s_geo.max(), 13) + 0.5, 0.5)
        n_, b_, parches = a2.hist(s_geo, bins=bins, color=VE, edgecolor="white", lw=0.6)
        for pa, izq in zip(parches, b_[:-1]):
            if izq >= s_max: pa.set_facecolor(CO)
        a2.axvline(s_max, color=TI, ls="--", lw=2); a2.text(s_max + 0.12, max(n_) * 1.12, f"límite {s_max:g} m", color=TI, fontsize=12, fontweight="bold")
        a2.set_ylim(0, max(n_) * 1.25)
        nc, nf = int((~falla).sum()), int(falla.sum())
        a2.text(-0.4, max(n_) * 1.06, f"cumple\n{nc} de {len(falla)}", color=VE, fontsize=15, fontweight="bold", ha="left")
        a2.text(8.0, max(n_) * 0.62, f"falla\n{nf} de {len(falla)} ({falla.mean():.0%})", color=CO, fontsize=15, fontweight="bold", ha="center")
        a2.set_xlabel("abatimiento del pozo más exigido [m], una barra por cada medio metro", fontsize=11); a2.set_ylabel("geologías nuevas", fontsize=11)
        a2.set_title("Fritz: en promedio, cada pozo cumple  ✓", color=TI, fontsize=15, fontweight="bold")
        for sp in ("top", "right"): a2.spines[sp].set_visible(False)
        fig.tight_layout(); plt.show()


def nube_decision(F_geo, F_modelo, frente=None, verde=None, ejes=("Abatimiento total [m]", "Costo [$/d]"), titulo=None, promedio=True, n_marcas=0):
    """Robustez de tipo I con una decisión (lámina «Solución robusta de tipo I»): sus resultados en cada geología (la nube B(x)), lo que prometía un
    solo modelo y el promedio, sobre el plano de los dos primeros objetivos. F_geo: (N, ≥2); F_modelo: (≥2,); frente y verde: (n, ≥2) opcionales."""
    import matplotlib.pyplot as plt
    F_geo = np.asarray(F_geo); prom = F_geo.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if frente is not None:
        ax.scatter(frente[:, 0], frente[:, 1], s=10, color="#333333", alpha=0.6, label="frente negro: un solo modelo")
    if verde is not None:
        ax.scatter(verde[:, 0], verde[:, 1], s=10, color="#2e9b4f", alpha=0.6, label="verde: el promedio de cada decisión")
    ax.scatter(F_geo[:, 0], F_geo[:, 1], s=16, color="#2E9BB5", alpha=0.45, label=f"B(x): la decisión en {len(F_geo)} geologías")
    ax.plot(F_modelo[0], F_modelo[1], "o", ms=12, mfc="none", mec=_ORO, mew=2.5, label=f"lo que prometía un solo modelo: ({F_modelo[0]:.1f} m, {F_modelo[1]:.0f} $/d)")
    if n_marcas:
        ax.scatter(F_geo[:n_marcas, 0], F_geo[:n_marcas, 1], s=60, color=_ORO, edgecolor="k", zorder=4, label=f"las geologías 0 a {n_marcas - 1} de la tabla")
    if promedio:
        ax.plot(prom[0], prom[1], "D", ms=11, mfc="#2e9b4f", mec="k", label=f"el promedio, f_eff(x): ({prom[0]:.1f} m, {prom[1]:.0f} $/d)")
    ax.set_xlabel(ejes[0]); ax.set_ylabel(ejes[1].replace("$", r"\$")); ax.legend(fontsize=9, loc="upper left")
    ax.set_title(titulo or "Una decisión, todas las geologías: su promedio es la función efectiva"); fig.tight_layout(); plt.show()


def decision_en_geologias(s_geologias, s_prometido, r_acum, error, limite=3.0, nombre="pozo 1"):
    """Una decisión en muchas geologías: el histograma de su abatimiento en un pozo (con el límite y lo que prometía el homogéneo)
    y el riesgo estimado con las primeras N geologías, con su banda de Monte Carlo."""
    import matplotlib.pyplot as plt
    N = np.arange(1, len(r_acum) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.3))
    ax1.hist(s_geologias, bins=30, color="#2E7DB5", edgecolor="k", alpha=0.8); ax1.axvline(limite, color=_ROJO, ls="--", lw=2, label=f"límite {limite:g} m")
    ax1.axvline(s_prometido, color=_AZUL, lw=2, label=f"lo que prometía el homogéneo: {s_prometido:.2f} m"); ax1.set_xlabel(f"abatimiento en el {nombre} [m]"); ax1.legend()
    ax1.set_title(f"La misma decisión, {len(s_geologias)} geologías")
    ax2.fill_between(N, r_acum - 2 * error, r_acum + 2 * error, color="#c9d6e4", label="± 2 errores estándar")
    ax2.plot(N, r_acum, color=_AZUL, lw=2, label="riesgo con las primeras N geologías"); ax2.set_ylim(0, 1)
    ax2.set_xlabel("número de geologías N"); ax2.set_ylabel("riesgo"); ax2.legend(); ax2.set_title("Monte Carlo, como en el ejemplo 1")
    fig.tight_layout(); plt.show()


def frente_agua_riesgo(FR, F_omo=None, referencia=None, demanda=1200.0, xmax=1000):
    """El frente exacto agua contra riesgo (déficit, riesgo en las columnas 0 y 1), los puntos de OMOPSO robusto y una decisión de referencia (agua, riesgo)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.step(demanda - FR[:, 0], FR[:, 1], where="post", color=_AZUL, lw=2.5, label="frente exacto (la malla en las geologías)")
    if F_omo is not None:
        ax.plot(demanda - F_omo[:, 0], F_omo[:, 1], "o", mfc=_ORO, mec="k", ms=6, label="OMOPSO robusto")
    if referencia is not None:
        ax.plot(referencia[0], referencia[1], "*", ms=20, mfc=_ROJO, mec="k", label=referencia[2] if len(referencia) > 2 else "óptimo de la sesión 1 (homogéneo)")
    ax.set_xlabel("bombeo total [m³/d]"); ax.set_ylabel("riesgo de incumplir"); ax.set_xlim(0, xmax); ax.legend()
    ax.set_title("Cuánta agua, con qué riesgo"); fig.tight_layout(); plt.show()



def mosaico_fallas(casos, titulo="El riesgo, geología por geología"):
    """Un mosaico de casillas por decisión: cada casilla es una geología, verde si la decisión cumple ahí y roja si rompe alguna regla.
    casos: lista de (texto, fallas) con fallas = arreglo de True/False, uno por geología. El riesgo es la fracción de casillas rojas."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    n = len(casos)
    fig, ejes = plt.subplots(1, n, figsize=(4.6 * n, 3.4))
    ejes = np.atleast_1d(ejes)
    for ax, (texto, fallas) in zip(ejes, casos):
        f = np.asarray(fallas, dtype=bool); cols = 20; filas = int(np.ceil(len(f) / cols))
        M = np.full(filas * cols, np.nan); M[:len(f)] = f
        ax.imshow(M.reshape(filas, cols), cmap=ListedColormap([_VERDE, _ROJO]), vmin=0, vmax=1)
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True); ax.set_yticks(np.arange(-0.5, filas, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="both", length=0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{texto}\nfalla en {f.sum()} de {len(f)}  ->  riesgo {f.mean():.1%}", fontsize=10.5)
    fig.suptitle(titulo + " (verde = cumple, rojo = rompe alguna regla)", fontsize=12); fig.tight_layout(); plt.show()


def malla_agua_riesgo(Z, R, FR, demanda=1200.0, niveles=(0.0, 0.05, 0.10, 0.20, 0.30), xmax=1000):
    """Todas las decisiones de la malla como puntos (agua total contra riesgo), la escalera del frente exacto y cinco niveles de riesgo marcados.
    Z: agua de cada decisión; R: su riesgo; FR: el frente (déficit, riesgo, índice) de frente_riesgo."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.scatter(Z, 100 * R, s=2, color="#9AA5AD", alpha=0.25, lw=0, label=f"las {len(Z):,} decisiones de la malla".replace(",", " "))
    agua = demanda - FR[:, 0]
    ax.step(agua, 100 * FR[:, 1], where="post", color=_AZUL, lw=2.5, label="frente exacto: la que más agua da para cada riesgo")
    for r in niveles:
        a = demanda - FR[FR[:, 1] <= r + 1e-12][-1][0]
        ax.plot(a, 100 * r, "o", ms=9, mfc=_ORO, mec="k", zorder=5)
        ax.annotate(f"riesgo ≤ {r:.0%}: {a:.0f} m³/d", (a, 100 * r), xytext=(10, -4), textcoords="offset points", fontsize=9.5,
                    color=_ORO, fontweight="bold", bbox=dict(fc="white", ec=_ORO, lw=0.8))
    ax.set_xlim(0, xmax); ax.set_ylim(-3, 103)
    ax.set_xlabel("bombeo total de la decisión [m³/d]"); ax.set_ylabel("riesgo: geologías en que rompe [%]")
    ax.set_title("Cada punto es una decisión: más agua, más riesgo"); ax.legend(loc="lower right", fontsize=9, markerscale=4)
    fig.tight_layout(); plt.show()


def malla_bombeo(paso=10.0, maximo=400.0, n_pozos=3):
    """Todas las decisiones de bombeo de una malla: cada pozo de 0 a `maximo` en pasos de `paso`. Con 0–400 cada 10 y tres pozos, 41³ = 68 921 filas (p1, p2, p3)."""
    g = np.arange(0, maximo + paso / 2, paso)
    return np.array(np.meshgrid(*([g] * n_pozos), indexing="ij")).reshape(n_pozos, -1).T


def fallas_malla(M, A, H, s_max=3.0, s_hum=1.25, lim_pozo=None, lim_hum=None, perdidas=None):
    """¿Rompe cada decisión de la malla M en cada geología? Tabla de True/False de (decisiones x geologías).
    A, H: matrices de respuesta de las geologías. Opcionales (apéndice A.4): lim_pozo y lim_hum, un límite por geología; perdidas, un abatimiento extra C·p² por geología."""
    n = len(A)
    lim_pozo = np.full(n, s_max) if lim_pozo is None else lim_pozo
    lim_hum = np.full(n, s_hum) if lim_hum is None else lim_hum
    F = np.zeros((len(M), n), dtype=bool)
    for k in range(n):
        s = M @ A[k].T
        if perdidas is not None:
            s = s + perdidas[k] * M**2
        F[:, k] = (s > lim_pozo[k] + 1e-9).any(axis=1) | ((M @ H[k]) > lim_hum[k] + 1e-9)
    return F


def frente_riesgo(Z, R, demanda=1200.0):
    """El frente exacto agua contra riesgo: para cada nivel de riesgo que aparece en R, la decisión con más agua Z. Una fila por escalón:
    (déficit = demanda − agua, riesgo, número de la decisión en la malla)."""
    filas = []
    for r in np.unique(R):
        ok = R <= r + 1e-12
        i = np.flatnonzero(ok)[np.argmax(Z[ok])]
        if not filas or demanda - Z[i] < filas[-1][0] - 1e-9:
            filas.append([demanda - Z[i], R[i], i])
    return np.array(filas)


def agua_con_riesgo(FR, r, demanda=1200.0):
    """El agua máxima del frente FR con riesgo ≤ r [m³/d]."""
    return demanda - FR[FR[:, 1] <= r + 1e-12][-1][0]


def decision_con_riesgo(FR, r):
    """El número (en la malla) de la decisión del frente FR con más agua y riesgo ≤ r."""
    return int(FR[FR[:, 1] <= r + 1e-12][-1][2])


def falta_contra_frente(F_omo, FR):
    """Cuánta agua le falta al frente de OMOPSO frente al exacto, nivel por nivel de riesgo (m³/d; 0 = la misma agua). Columnas: (déficit, riesgo)."""
    return np.array([F_omo[F_omo[:, 1] <= r + 1e-12, 0].min() - d for d, r, _ in FR if (F_omo[:, 1] <= r + 1e-12).any()])

def riesgo_contra_N(filas):
    """El riesgo real (en geologías nuevas) contra el número de geologías con que se optimizó. filas: (N, riesgo pedido, agua, riesgo real)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pedidos = sorted({f_[1] for f_ in filas}); colores = [_AZUL, _ORO, _ROJO]
    for r, c in zip(pedidos, colores):
        xs_n = [f_[0] for f_ in filas if f_[1] == r]; ys = [f_[3] for f_ in filas if f_[1] == r]
        ax.plot(xs_n, ys, "o-", color=c, lw=2, label=f"pidiendo riesgo ≤ {r:.0%}"); ax.axhline(r, color=c, ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("geologías con que se optimizó (N)"); ax.set_ylabel("riesgo real en 200 geologías nuevas"); ax.legend()
    ax.set_title("Con pocas geologías, el optimizador se engaña"); fig.tight_layout(); plt.show()


def abatimiento_maximo(A, decisiones, limite=3.0):
    """Para cada decisión (p, título), el histograma del abatimiento del pozo más exigido en las geologías del ensamble A (n, 3, 3)."""
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, len(decisiones), figsize=(6.5 * len(decisiones), 4.5), sharex=True)
    for ax, (p, titulo) in zip(np.atleast_1d(ejes), decisiones):
        s_max = (A @ np.asarray(p, float)).max(axis=1)
        ax.hist(s_max, bins=30, color="#2E7DB5", edgecolor="k", alpha=0.8); ax.axvline(limite, color=_ROJO, ls="--", lw=2)
        ax.set_xlabel("abatimiento del pozo más exigido [m]"); ax.set_title(f"{titulo}: rompe en {np.mean(s_max > limite + 1e-9):.0%}")
    fig.tight_layout(); plt.show()


def escalera_costo(frentes_fuente, escalones, demanda=1200.0, determinista=807.34, riesgo=0.05):
    """Izquierda: el frente agua contra riesgo de cada fuente de incertidumbre. Derecha: la escalera del costo de la robustez (agua con riesgo ≤ 5 %)."""
    import matplotlib.pyplot as plt
    colores = [_AZUL, _ORO, "#7a9cc6", _ROJO, _VERDE]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for k, (nombre, Fq) in enumerate(frentes_fuente.items()):
        ax1.step(demanda - Fq[:, 0], Fq[:, 1], where="post", lw=2.2, color=colores[k % len(colores)], label=nombre)
    ax1.axvline(determinista, color="k", ls=":", label="determinista (homogéneo)")
    ax1.set_xlabel("bombeo total [m³/d]"); ax1.set_ylabel("riesgo"); ax1.set_xlim(0, 900); ax1.set_ylim(0, 0.5); ax1.legend(fontsize=9)
    ax1.set_title("Cada incertidumbre corre el frente a la izquierda")
    ax2.bar([e[0] for e in escalones], [e[1] for e in escalones], color=[_GRIS, _AZUL, _ORO, _ROJO][:len(escalones)], edgecolor="k")
    for k, (nombre, z) in enumerate(escalones):
        ax2.text(k, z + 15, f"{z:.0f}", ha="center", fontsize=12, fontweight="bold")
        if k:
            ax2.annotate(f"−{escalones[k - 1][1] - z:.0f}", (k - 0.5, (escalones[k - 1][1] + z) / 2), ha="center", color=_ROJO, fontsize=12, fontweight="bold")
    ax2.set_ylabel(f"agua con riesgo ≤ {riesgo:.0%} [m³/d]"); ax2.set_title("El costo de la robustez, en metros cúbicos")
    fig.tight_layout(); plt.show()


def valor_prueba(agua_post, z_prior, titulo="El valor de una prueba de bombeo"):
    """El agua que se decide bombear con la información de una prueba, en cada geología «real», contra la decisión sin la prueba."""
    import matplotlib.pyplot as plt
    agua_post = np.asarray(agua_post)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(agua_post, bins=20, color="#8fd18f", edgecolor="k", label=f"con la prueba ({len(agua_post)} geologías reales)")
    ax.axvline(z_prior, color=_AZUL, lw=2.5, label=f"sin la prueba: {z_prior:.0f} m³/d")
    ax.axvline(agua_post.mean(), color=_ORO, lw=2.5, ls="--", label=f"con la prueba, en promedio: {agua_post.mean():.0f} m³/d")
    ax.set_xlabel("agua que se decide bombear [m³/d]"); ax.set_ylabel("casos"); ax.legend(fontsize=9); ax.set_title(titulo); fig.tight_layout(); plt.show()


# ============================================================== objetivo único (Sesión 4, secciones 2.1 y 3.1)
def funcion_efectiva_unica(valores, f_hom, nombre="la decisión de la sesión 1", xl="f(p, X): agua menos multa [m³/d]"):
    """Objetivo único: los valores de f de una decisión en cada geología (histograma), lo que prometía el homogéneo y su promedio (la función efectiva).
    Derecha: el promedio con las primeras N geologías."""
    import matplotlib.pyplot as plt
    v = np.asarray(valores, dtype=float); N = np.arange(1, len(v) + 1); prom = v.mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a1.hist(v, bins=30, color="#2E7DB5", edgecolor="k", alpha=0.8, label=f"f en cada una de las {len(v)} geologías")
    a1.axvline(f_hom, color=_ORO, lw=3, label=f"lo que prometía el homogéneo: {f_hom:,.0f}".replace(",", " "))
    a1.axvline(prom, color=_VERDE, lw=3, ls="--", label=f"el promedio, f_eff: {prom:,.0f}".replace(",", " "))
    a1.set_xlabel(xl); a1.set_ylabel("geologías"); a1.legend(fontsize=9, loc="upper left"); a1.set_title(f"{nombre}: un número por geología", fontsize=11)
    a2.plot(N, np.cumsum(v) / N, color=_VERDE, lw=2.2); a2.axhline(f_hom, color=_ORO, lw=2)
    a2.set_xlabel("número de geologías N"); a2.set_ylabel("promedio con las primeras N"); a2.set_title("La integral, paso a paso: sumar y dividir entre N", fontsize=11)
    a2.grid(alpha=0.25); fig.tight_layout(); plt.show()


def cubo_decisiones(f, A, H, p_s1, p_linea, p_enjambre, paso=20, pmax=400, alto=620):
    """«La cima se mueve», en 3-D (2026-11-23, ARP): el cubo de las decisiones (bombeo de los tres pozos, de 0 a pmax cada paso m³/d), interactivo (plotly).
    Cada punto es una decisión calificada con el promedio en las geologías, f(p, A, H).mean(); en color las que califican ≥ 0 y en gris tenue las que pagan
    más multa que agua. Encima: la línea (la decisión de la sesión 1 escalada de 0 a 1.1), la de la sesión 1, la mejor de la línea y la del enjambre."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as e:
        raise ImportError("Falta plotly (la figura 3-D interactiva):  pip install -r requirements.txt") from e
    if "google.colab" not in sys.modules:
        pio.renderers.default = "plotly_mimetype"
    g = np.arange(0, pmax + 1e-9, paso)
    P = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T          # todas las decisiones de la malla
    fe = np.array([f(p, A, H).mean() for p in P])                               # su calificación promedio
    k = int(np.argmax(fe))
    print(f"{len(P)} decisiones en la malla; la mejor: {np.round(P[k]).astype(int)} = {P[k].sum():.0f} m³/d, calificación {fe[k]:.0f}")
    print(f"califican ≥ 0 (dan más agua que multa): {np.sum(fe >= 0)} de {len(P)}")
    buenas = fe >= 0
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=P[~buenas, 0], y=P[~buenas, 1], z=P[~buenas, 2], mode="markers", name="pagan más multa que agua",
                               marker=dict(size=2, color="#9aa5b1", opacity=0.25), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=P[buenas, 0], y=P[buenas, 1], z=P[buenas, 2], mode="markers", name="califican ≥ 0",
                               marker=dict(size=4, color=fe[buenas], colorscale="Viridis", colorbar=dict(title="calificación<br>promedio"), opacity=0.85),
                               text=[f"({a:.0f}, {b:.0f}, {c:.0f}) → {v:.0f}" for (a, b, c), v in zip(P[buenas], fe[buenas])], hoverinfo="text"))
    lin = np.outer([0, 1.1], np.asarray(p_s1, float))
    fig.add_trace(go.Scatter3d(x=lin[:, 0], y=lin[:, 1], z=lin[:, 2], mode="lines", name="la línea: la sesión 1 escalada", line=dict(color="black", width=5)))
    def esq(p): return "(" + ", ".join(f"{v:.0f}" for v in np.asarray(p, float)) + ")"
    for p, nombre, corto, col, sim in ((p_s1, f"sesión 1: {esq(p_s1)} = {np.sum(p_s1):.0f} m³/d (un solo modelo)", "sesión 1", "goldenrod", "diamond"),
                                       (p_linea, f"mejor de la línea: {esq(p_linea)} = {np.sum(p_linea):.0f} m³/d (mismo reparto)", "mejor de la línea", "green", "circle"),
                                       (p_enjambre, f"enjambre (3.1): {esq(p_enjambre)} = {np.sum(p_enjambre):.0f} m³/d (pozo 2 apagado)", "enjambre", "red", "diamond")):
        p = np.asarray(p, float)
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name=nombre, hovertext=[f"{corto} {esq(p)}"], hoverinfo="text",
                                   marker=dict(size=9, color=col, symbol=sim, line=dict(color="black", width=2))))
    fig.update_layout(title="El cubo de las decisiones, calificadas con el promedio en 200 geologías · gíralo con el mouse", height=alto,
                      margin=dict(l=0, r=0, t=40, b=0), legend=dict(x=0, y=1, font=dict(size=13)),
                      scene=dict(xaxis_title="pozo 1 [m³/d]", yaxis_title="pozo 2 [m³/d]", zaxis_title="pozo 3 [m³/d]",
                                 camera=dict(eye=dict(x=-1.55, y=-1.75, z=0.95))))
    fig.show()


def cima_se_mueve(z, f_hom, f_eff, linea=None, ymin=-1000):
    """«La cima se mueve» (2026-11-23, ARP): solo dibuja. z = agua de cada decisión de la línea; f_hom = su calificación con un solo modelo; f_eff = su promedio
    en las geologías. Eje vertical recortado en ymin; donde f_eff se sale, una flecha dice cuánto vale en el máximo de un solo modelo."""
    import matplotlib.pyplot as plt
    z, f_hom, f_eff = np.asarray(z, float), np.asarray(f_hom, float), np.asarray(f_eff, float)
    kh, ke = int(np.argmax(f_hom)), int(np.argmax(f_eff))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(z, f_hom, color=_ORO, lw=3, label="un solo modelo (el homogéneo)")
    dentro = f_eff >= ymin                                       # la parte que cabe; lo demás lo dice la flecha
    ax.plot(z[dentro], f_eff[dentro], color=_VERDE, lw=3, ls="--", label="el promedio en las 200 geologías (f_eff)")
    def esquema(k):                                              # (p1, p2, p3) de la decisión k, si se dio la línea
        return "" if linea is None else "\n(" + ", ".join(f"{v:.0f}" for v in np.asarray(linea)[k]) + ") m³/d"
    ax.plot(z[kh], f_hom[kh], "o", ms=13, color=_ORO, mec="k", zorder=5)
    ax.annotate(f"un solo modelo:\nlo mejor es bombear {z[kh]:.0f} m³/d{esquema(kh)}", (z[kh], f_hom[kh]), xytext=(-20, 10), textcoords="offset points", ha="right",
                fontsize=11, color=_ORO, fontweight="bold")
    ax.plot(z[ke], f_eff[ke], "o", ms=13, color=_VERDE, mec="k", zorder=5)
    ax.annotate(f"el promedio:\nlo mejor es bombear {z[ke]:.0f} m³/d{esquema(ke)}", (z[ke], f_eff[ke]), xytext=(10, 22), textcoords="offset points",
                fontsize=11, color=_VERDE, fontweight="bold")
    if linea is not None:                                        # el reparto, que es el mismo en toda la línea
        r = np.asarray(linea)[-1] / np.asarray(linea)[-1].sum()
        ax.text(0.68, 0.45, "en toda la línea, el mismo reparto:\n" + "\n".join(f"pozo {j + 1}: {100 * r[j]:.0f} %" for j in range(3)),
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5, bbox=dict(fc="white", ec="0.6", boxstyle="round,pad=0.4"))
    ax.annotate(f"sigue bajando: en {z[kh]:.0f} m³/d\nel promedio vale {f_eff[kh]:,.0f}".replace(",", " "), (z[kh], ymin), xytext=(z[kh] - 330, ymin + 260),
                fontsize=10.5, color=_VERDE, arrowprops=dict(arrowstyle="->", color=_VERDE, lw=1.5))
    ax.set_ylim(ymin, max(f_hom) * 1.5); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("agua bombeada [m³/d]  (la decisión de la sesión 1 escalada: menos agua a la izquierda)")
    ax.set_ylabel("calificación: agua menos multa")
    ax.set_title("La cima se mueve: con el promedio, a partir de unos cuantos m³/d la multa crece más rápido que el agua", fontsize=11.5)
    ax.legend(fontsize=10, loc="upper left"); ax.grid(alpha=0.25)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); plt.show()


def curva_efectiva(t, f_hom, f_eff, z, titulo="A lo largo de la decisión de la sesión 1: t × (370.87, 136.47, 300)"):
    """Objetivo único: f con el homogéneo y la función efectiva a lo largo de una línea de decisiones (t veces una decisión); z = agua de cada una."""
    import matplotlib.pyplot as plt
    t = np.asarray(t); f_hom = np.asarray(f_hom); f_eff = np.asarray(f_eff)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(t, f_hom, color=_ORO, lw=2.8, label="f con el homogéneo (un solo modelo)")
    ax.plot(t, f_eff, color=_VERDE, lw=2.8, ls="--", label="f_eff: el promedio en las 200 geologías")
    for f, col, txt, off in ((f_hom, _ORO, "óptimo con un modelo", (-175, -45)), (f_eff, _VERDE, "óptimo robusto", (12, 14))):
        k = int(np.argmax(f)); ax.plot(t[k], f[k], "o", ms=12, color=col, mec="k", zorder=5)
        ax.annotate(f"{txt}\nt = {t[k]:.2f}: {z[k]:.0f} m³/d", (t[k], f[k]), xytext=off, textcoords="offset points", fontsize=10, color=col, fontweight="bold")
    lo = min(np.max(f_eff) - 1.2 * (np.max(f_hom) - np.min(f_eff[: len(t) // 2 + 1])), -200)
    ax.set_ylim(max(lo, np.min(f_eff)), np.max(f_hom) * 1.25); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("t (fracción de la decisión de la sesión 1)"); ax.set_ylabel("agua menos multa [m³/d]"); ax.legend(fontsize=9, loc="lower left")
    ax.set_title(titulo, fontsize=11); ax.grid(alpha=0.25); fig.tight_layout(); plt.show()


# ============================================================== OCA (Sesión 4, sección 5): zonas de protección y geologías representativas
def zonas_proteccion(logK, semilla=10_000, verbose=True):
    """La zona de protección (WHPA) de los tres pozos en cada geología, con el simulador del curso (rrwpt: flujo con elementos finitos y
    caminata aleatoria inversa de partículas desde cada pozo; tiempo de tránsito de 30 días). Devuelve un arreglo (N, 91, 91): para cada
    nodo, el porcentaje de partículas que lo cruzan en el tiempo de tránsito (≥ 50 %: el nodo está en la zona de esa geología).
    Unos 0.45 s por geología la primera vez; se guarda en _cache_hidra/ (el nombre lleva una huella del ensamble)."""
    import hashlib
    import time
    logK = np.asarray(logK); n = len(logK)
    huella = hashlib.md5(np.ascontiguousarray(logK[: min(n, 3)]).tobytes()).hexdigest()[:10]
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache_hidra")
    ruta = os.path.join(carpeta, f"zonas_{huella}_{n}.npz")
    if os.path.exists(ruta):
        return np.load(ruta)["zonas"]
    e = _preparar(); grid = e["grid"]
    d = grid.data.d_pts; npt = tuple(int(v) for v in grid.data.n_pts)
    yy, xx = np.meshgrid(np.arange(npt[0]) * d[0], np.arange(npt[1]) * d[1], indexing="ij")
    junto_pozo = np.zeros(npt, dtype=bool)
    for x, y in pozos_xy():                                         # el pozo capta siempre su entorno inmediato (las partículas nacen a su alrededor)
        junto_pozo |= (xx - x) ** 2 + (yy - y) ** 2 <= 75.0 ** 2
    zonas = np.zeros((n,) + npt, dtype=np.uint8); t0 = time.time()
    if verbose:
        print(f"Delineando la zona de protección en {n} geologías con el simulador (unos {0.45 * n:.0f} s la primera vez)…", flush=True)
    for k in range(n):
        yk = np.asarray(logK[k], dtype=float).reshape(-1, order="F")
        r = _pipeline.run_realization(e["ctrl"], grid, e["model"], e["flowpar"], rng=np.random.default_rng(semilla + k), verbose=False, yk=yk)
        m = np.asarray(_pipeline.prob_map_2d(r, grid), dtype=float); m[junto_pozo] = 100.0
        zonas[k] = np.round(m).astype(np.uint8)
        if verbose and (k + 1) % 50 == 0:
            print(f"  geología {k + 1} de {n} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(carpeta, exist_ok=True); np.savez_compressed(ruta, zonas=zonas)
    return zonas


def _ejes_mapa(ax, titulo=None):
    for x, y in pozos_xy():
        ax.plot(x, y, "^", ms=10, color="#D9534F", mec="k")
    ax.set_xlim(0, LADO); ax.set_ylim(0, LADO); ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    if titulo:
        ax.set_title(titulo, fontsize=11)


def zona_una_geologia(logK_k, zona_k, k=0):
    """Una geología y su zona de protección: el campo de ln K con los pozos, y la zona (nodos con ≥ 50 % de las partículas) encima."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6))
    im = a1.imshow(logK_k, origin="lower", extent=[0, LADO, 0, LADO], cmap="viridis", vmin=-8.5, vmax=-4.5)
    fig.colorbar(im, ax=a1, shrink=0.8, label="ln K"); _ejes_mapa(a1, f"Geología {k}: el campo de ln K y los tres pozos")
    a2.imshow(logK_k, origin="lower", extent=[0, LADO, 0, LADO], cmap="Greys", vmin=-9.5, vmax=-3.5, alpha=0.55)
    dentro = np.ma.masked_where(zona_k < 50, np.ones_like(zona_k, dtype=float))
    a2.imshow(dentro, origin="lower", extent=[0, LADO, 0, LADO], cmap="autumn", alpha=0.65, vmin=0, vmax=1)
    xs = np.linspace(0, LADO, zona_k.shape[1]); a2.contour(xs, xs, zona_k, levels=[50], colors="#B03A2E", linewidths=1.8)
    _ejes_mapa(a2, f"Su zona de protección (30 días): {np.sum(zona_k >= 50) * 225 / 1e6:.2f} km²")
    fig.tight_layout(); plt.show()


def mapa_probabilidad_zonas(P, niveles=(0.1, 0.5, 0.9), P_sub=None, nombre_sub="subconjunto", errores=None):
    """El mapa de probabilidad de captura del ensamble (P: un valor por nodo) con sus isolíneas. Con P_sub, a la derecha se comparan las
    isolíneas del ensamble (negras) con las del subconjunto (rojas a trazos) y, si se dan, los errores de cada isolínea."""
    import matplotlib.pyplot as plt
    P = np.asarray(P).reshape(N_CELDAS + 1, N_CELDAS + 1); xs = np.linspace(0, LADO, P.shape[1])
    colores = ("#F2F2F2", "#F0AD4E", "#D9534F")
    fig, axs = plt.subplots(1, 2 if P_sub is not None else 1, figsize=(13 if P_sub is not None else 7, 5.6), squeeze=False)
    a = axs[0, 0]; im = a.imshow(100 * P, origin="lower", extent=[0, LADO, 0, LADO], cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(im, ax=a, shrink=0.8, label="probabilidad de captura [%]")
    cs = a.contour(xs, xs, P, levels=list(niveles), colors=["#1B2B33", "#C08A0B", "#B03A2E"], linewidths=1.8)
    a.clabel(cs, fmt=lambda v: f"{100 * v:.0f} %", fontsize=9); _ejes_mapa(a, "Todas las geologías: la probabilidad de captura y sus isolíneas")
    if P_sub is not None:
        b = axs[0, 1]; Ps = np.asarray(P_sub).reshape(P.shape)
        b.contour(xs, xs, P, levels=list(niveles), colors="k", linewidths=1.8)
        b.contour(xs, xs, Ps, levels=list(niveles), colors="#D9534F", linewidths=1.6, linestyles="--")
        b.plot([], [], "k-", label="todas las geologías"); b.plot([], [], "--", color="#D9534F", label=nombre_sub)
        if errores is not None:
            for t, e in zip(niveles, errores):
                b.plot([], [], " ", label=f"isolínea {100 * t:.0f} %: {e:.0%} mal")
        b.legend(fontsize=9, loc="lower right"); _ejes_mapa(b, f"Isolíneas: todas contra {nombre_sub}")
    fig.tight_layout(); plt.show()


def mapa_cumplir(DEC, P, niveles=(1.0, 0.9, 0.8), cortes=(0, 60, 120), P_sub=None, nombre_sub="subconjunto"):
    """El «mapa» de Fritz: la probabilidad de que cada decisión de referencia cumpla todas las reglas, en tres cortes del pozo 3, con las
    isolíneas de 100, 90 y 80 % (riesgo 0, 10 y 20 %). Con P_sub, las isolíneas del subconjunto encima, a trazos."""
    import matplotlib.pyplot as plt
    DEC = np.asarray(DEC); P = np.asarray(P); g1 = np.unique(DEC[:, 0]); g2 = np.unique(DEC[:, 1])
    fig, axs = plt.subplots(1, len(cortes), figsize=(5 * len(cortes), 4.6))
    colores = ["#1F6FB2", "#C08A0B", "#B03A2E"]
    def rejilla(sel, valores):
        M = np.full((len(g2), len(g1)), np.nan)
        for (p1, p2), v in zip(DEC[sel, :2], valores[sel]):
            M[np.searchsorted(g2, p2), np.searchsorted(g1, p1)] = v
        return M
    for a, p3 in zip(np.atleast_1d(axs), cortes):
        sel = DEC[:, 2] == p3; M = rejilla(sel, P)
        d1, d2 = (g1[1] - g1[0]) / 2, (g2[1] - g2[0]) / 2
        im = a.imshow(100 * M, origin="lower", extent=[g1[0] - d1, g1[-1] + d1, g2[0] - d2, g2[-1] + d2], cmap="Greens", vmin=0, vmax=100, aspect="auto", alpha=0.7)
        a.contour(g1, g2, M, levels=sorted(t - 1e-6 for t in niveles), colors=colores[::-1][: len(niveles)], linewidths=2.2)
        if P_sub is not None:
            a.contour(g1, g2, rejilla(sel, np.asarray(P_sub)), levels=sorted(t - 1e-6 for t in niveles), colors=colores[::-1][: len(niveles)],
                      linewidths=1.6, linestyles="--")
        a.set_xlabel("bombeo del pozo 1 [m³/d]"); a.set_ylabel("bombeo del pozo 2 [m³/d]"); a.set_title(f"pozo 3 = {p3:g} m³/d", fontsize=11)
    for t, c in zip(niveles, colores):
        axs[-1].plot([], [], "-", color=c, lw=2, label=f"cumple en {100 * t:.0f} % (riesgo {100 * (1 - t):.0f} %)")
    if P_sub is not None:
        axs[-1].plot([], [], "k--", label=nombre_sub)
    axs[-1].legend(fontsize=8.5, loc="upper right")
    fig.subplots_adjust(wspace=0.32); fig.colorbar(im, ax=axs, shrink=0.85, label="probabilidad de cumplir todas las reglas [%]"); plt.show()


def espacio_rasgos(RAS, elegidas=None, ejemplos=(), nombres=("distancia del campo de K al medio", "«área» de la zona", "distancia de la zona a la media"),
                   titulo="Cada geología es un punto en el espacio de tres rasgos"):
    """Las geologías como puntos en el espacio de tres rasgos (normalizados entre 0 y 1); en rojo, las elegidas; en oro, algunos ejemplos."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 5.2)); RAS = np.asarray(RAS)
    a = fig.add_subplot(1, 2, 1, projection="3d")
    a.scatter(*RAS.T, s=14, color="#7C8B99", alpha=0.55, label=f"las {len(RAS)} geologías")
    if elegidas is not None:
        a.scatter(*RAS[elegidas].T, s=70, color="#D9534F", edgecolor="k", label=f"las {len(elegidas)} elegidas", depthshade=False)
    for k in ejemplos:
        a.scatter(*RAS[k], s=110, color="#C08A0B", edgecolor="k", depthshade=False); a.text(*RAS[k], f"  {k}", fontsize=9)
    a.set_xlabel(nombres[0], fontsize=8); a.set_ylabel(nombres[1], fontsize=8); a.set_zlabel(nombres[2], fontsize=8)
    a.legend(fontsize=8, loc="upper left"); a.set_title(titulo, fontsize=10.5)
    b = fig.add_subplot(1, 2, 2)
    b.scatter(RAS[:, 1], RAS[:, 2], s=14, color="#7C8B99", alpha=0.6)
    if elegidas is not None:
        b.scatter(RAS[elegidas, 1], RAS[elegidas, 2], s=70, color="#D9534F", edgecolor="k")
    for k in ejemplos:
        b.scatter(RAS[k, 1], RAS[k, 2], s=110, color="#C08A0B", edgecolor="k"); b.annotate(f" geología {k}", RAS[k, 1:], fontsize=9)
    b.set_xlabel(nombres[1]); b.set_ylabel(nombres[2]); b.set_title("Visto desde un lado: los dos rasgos de la zona", fontsize=10.5); b.grid(alpha=0.25)
    fig.tight_layout(); plt.show()


def oca_contra_azar(err_azar, err_oca, tam, niveles_txt=("100 %", "90 %", "80 %"), err_primeras=None, titulo="¿Qué tan bien reproduce cada subconjunto las isolíneas?"):
    """El error de cada isolínea (fracción de su tamaño) para muchos subconjuntos al azar (histograma), la OCA (línea roja) y, si se da, las primeras K."""
    import matplotlib.pyplot as plt
    err_azar = np.asarray(err_azar) / np.asarray(tam); err_oca = np.asarray(err_oca) / np.asarray(tam)
    fig, axs = plt.subplots(1, len(tam), figsize=(4.6 * len(tam), 3.9))
    for j, a in enumerate(axs):
        a.hist(100 * err_azar[:, j], bins=20, color="#AEB9C4", edgecolor="k", lw=0.4, label=f"{len(err_azar)} subconjuntos al azar")
        a.axvline(100 * err_oca[j], color="#D9534F", lw=3, label=f"OCA: {err_oca[j]:.0%}")
        if err_primeras is not None:
            a.axvline(100 * err_primeras[j] / tam[j], color="#1B2B33", lw=1.8, ls="--", label=f"las primeras: {err_primeras[j] / tam[j]:.0%}")
        mejor = np.mean(err_azar[:, j] < err_oca[j])
        a.set_title(f"isolínea {niveles_txt[j]} · el azar le gana {mejor:.0%} de las veces", fontsize=10)
        a.set_xlabel("mal clasificadas [% de la isolínea]"); a.set_ylabel("subconjuntos"); a.legend(fontsize=8.5)
    fig.suptitle(titulo, fontsize=12); fig.tight_layout(); plt.show()


def oca_validacion(filas, niveles=(0.0, 0.10, 0.20)):
    """Lo que se pide y lo que se obtiene: el riesgo REAL (en geologías nuevas) de la decisión elegida con cada conjunto, para cada riesgo pedido.
    filas: [(nombre, (agua por nivel), (riesgo real por nivel), color)]."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6)); x = np.arange(len(niveles)); w = 0.8 / len(filas)
    for j, (nombre, agua, real, color) in enumerate(filas):
        xx = x + (j - (len(filas) - 1) / 2) * w
        a1.bar(xx, 100 * np.asarray(real), w, color=color, edgecolor="k", lw=0.4, label=nombre)
        a2.bar(xx, agua, w, color=color, edgecolor="k", lw=0.4, label=nombre)
        for xi, v in zip(xx, real):
            a1.text(xi, 100 * v + 0.6, f"{100 * v:.0f}", ha="center", fontsize=8)
        for xi, v in zip(xx, agua):
            a2.text(xi, v + 4, f"{v:.0f}", ha="center", fontsize=8)
    for xi, r in zip(x, niveles):
        a1.plot([xi - 0.45, xi + 0.45], [100 * r] * 2, ":", color="#B03A2E", lw=2)
    a1.plot([], [], ":", color="#B03A2E", lw=2, label="lo que se pidió")
    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels([f"se pide riesgo ≤ {100 * r:.0f} %" for r in niveles]); a.grid(axis="y", alpha=0.25)
    a1.set_ylabel("riesgo REAL en 200 geologías nuevas [%]"); a1.set_title("¿Cumple la decisión lo que promete?"); a1.legend(fontsize=8.5)
    a2.set_ylabel("agua de la decisión elegida [m³/d]"); a2.set_title("¿Cuánta agua se decide bombear?")
    fig.tight_layout(); plt.show()


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
