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
        ax.axvspan(c - 2 * sigma, c + 2 * sigma, color=_ORO, alpha=0.18, label="x ± 2σ: donde puede caer x + δ" if k == 0 else None)
        ax.plot(c, f(c), "*", ms=16, mfc=_ORO, mec="k")
        ax.annotate(f"x = {c:g}, f = {f(c):.2f}", (c, f(c)), xytext=(8, -4), textcoords="offset points", fontsize=10)
    ax.set_xlabel("x (la decisión)"); ax.set_ylabel("f(x)"); ax.set_ylim(0, 1.15); ax.legend(loc="upper right")
    ax.set_title(f"Dos cimas y la incertidumbre (σ = {sigma}): la angosta cabe dentro del error"); fig.tight_layout(); plt.show()


def esperanza_contra_f(xs, fx, Ex, sigma=None):
    """f(x) contra su valor esperado E[f(x + δ)] y los dos óptimos: el determinista (el máximo de f) y el robusto (el máximo de E)."""
    import matplotlib.pyplot as plt
    xs, fx, Ex = np.asarray(xs), np.asarray(fx), np.asarray(Ex)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, fx, color=_AZUL, lw=2.5, label="f(x): sin incertidumbre")
    ax.plot(xs, Ex, color=_ORO, lw=2.5, label="E[f(x + δ)]: con incertidumbre" + (f" (σ = {sigma})" if sigma is not None else ""))
    ax.plot(xs[np.argmax(fx)], fx.max(), "*", ms=18, mfc=_AZUL, mec="k", label=f"óptimo determinista: x = {xs[np.argmax(fx)]:.1f}")
    ax.plot(xs[np.argmax(Ex)], Ex.max(), "*", ms=18, mfc=_ORO, mec="k", label=f"óptimo robusto: x = {xs[np.argmax(Ex)]:.1f}")
    ax.set_xlabel("x"); ax.set_ylabel("valor"); ax.legend(); ax.set_title("La cima angosta no resiste la incertidumbre")
    fig.tight_layout(); plt.show()


def error_montecarlo(N, curvas):
    """La media acumulada con las primeras N muestras y su banda de ± 2 errores estándar. curvas: {etiqueta: (media_acumulada, error)}."""
    import matplotlib.pyplot as plt
    colores = [_AZUL, _ORO, _ROJO, _VERDE]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for k, (etiqueta, (media, error)) in enumerate(curvas.items()):
        c = colores[k % len(colores)]
        ax.fill_between(N, media - 2 * error, media + 2 * error, color=c, alpha=0.2)
        ax.plot(N, media, color=c, lw=2, label=f"x = {etiqueta}" if not isinstance(etiqueta, str) else etiqueta)
    ax.set_xscale("log"); ax.set_xlabel("número de muestras N"); ax.set_ylabel("E[f(x + δ)] estimada"); ax.legend()
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
    ax.plot(xs, Ex, color="#c9d6e4", lw=3, label="E[f(x + δ)] de todas las decisiones")
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


def mapas_ensamble(logK, n=10, vmin=-8.5, vmax=-4.5, columnas=5):
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
        ax.set_title(f"geología {k}"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ejes, label="ln K", shrink=0.6)
    plt.show()


def histograma_riesgo(riesgos, riesgo_max=None, titulo="¿Qué tan seguido falla cada frente?", xl="riesgo en 200 geologías de validación"):
    """Histogramas superpuestos del riesgo de las decisiones de cada frente. riesgos: {nombre: (valores, color)}; riesgo_max dibuja una línea."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    for nombre, (r, color) in riesgos.items():
        ax.hist(np.asarray(r), bins=np.linspace(0, 1, 21), color=color, alpha=0.6, label=f"{nombre} (media {np.mean(r):.0%})", edgecolor="k")
    if riesgo_max is not None:
        ax.axvline(riesgo_max, color="#1f5fbf", ls="--", lw=2, label=f"riesgo pedido: {riesgo_max:.0%}")
    ax.set_xlabel(xl); ax.set_ylabel("decisiones"); ax.legend(fontsize=9); ax.set_title(titulo); fig.tight_layout(); plt.show()


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
