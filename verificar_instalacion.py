"""verificar_instalacion.py — Comprueba que la computadora está lista para el curso.

Ejecutar desde la carpeta Curso_Opt con el entorno virtual activado:
    python verificar_instalacion.py

Revisa: versión de Python, paquetes (incluido otter, el calificador de las tareas),
estructura de carpetas y una corrida del simulador a través de hidra (fabrica la
matriz de respuesta del acuífero del curso y resuelve el problema de la Sesión 1). Si todo sale bien, imprime
«INSTALACIÓN CORRECTA» y el óptimo de referencia Z* = 969.92 m³/d.
"""
import importlib
import os
import sys
import time

OK, FALLO = "  [OK]  ", "  [X]   "
errores = 0


def reporta(cond, msg_ok, msg_fallo):
    global errores
    if cond:
        print(OK + msg_ok)
    else:
        print(FALLO + msg_fallo)
        errores += 1


print("\n=== 1. Python ===")
v = sys.version_info
reporta((3, 11) <= (v.major, v.minor) <= (3, 14),
        f"Python {v.major}.{v.minor}.{v.micro}",
        f"Python {v.major}.{v.minor}: el simulador del curso existe para 3.11 a 3.14; se recomienda 3.13")
en_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
reporta(en_venv, "Entorno virtual activo (.venv)",
        "No hay entorno virtual activo: ejecuta  .venv\\Scripts\\activate")

print("\n=== 2. Paquetes ===")
for pkg, nombre in [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
                    ("PIL", "pillow"), ("plotly", "plotly"), ("IPython", "jupyter / ipykernel"), ("otter", "otter-grader")]:
    try:
        m = importlib.import_module(pkg)
        reporta(True, f"{nombre} {getattr(m, '__version__', '')}", "")
    except ImportError:
        reporta(False, "", f"falta {nombre}:  pip install -r requirements.txt")
try:
    import pyamg
    print(OK + f"pyamg {pyamg.__version__} (solver rápido)")
except ImportError:
    print("  [--]   pyamg no instalado: el simulador funciona igual, solo más lento")

print("\n=== 3. Carpetas del curso ===")
raiz = os.path.dirname(os.path.abspath(__file__))
for nombre in ["Session_1", "Session_2", "Session_3", "Session_4", "hidra.py",
               os.path.join("Session_2", "Información", "codigo", "pso.py"),
               os.path.join("Session_3", "Información", "codigo", "pyOMOPSO"),
               os.path.join("Session_4", "Información", "codigo", "pyOMOPSO")]:
    reporta(os.path.exists(os.path.join(raiz, nombre)), nombre,
            f"no se encuentra «{nombre}» junto a este script — descomprime el ZIP completo")

print("\n=== 4. Simulador (acuífero del curso) ===")
if errores == 0:
    try:
        sys.path.insert(0, raiz)
        import numpy as np
        from scipy.optimize import linprog
        import hidra as ah
        t0 = time.time()
        alpha = ah.matriz_respuesta()
        dt = time.time() - t0
        print(OK + f"matriz de respuesta fabricada en {dt:.1f} s")
        b2 = ah.S_MAX - alpha[:, 2] * ah.P3_COMPROMETIDO
        res = linprog(c=[-1, -1], A_ub=alpha[:, :2], b_ub=b2,
                      bounds=[(0, 1000), (0, 1000)], method="highs")
        Z = -res.fun + ah.P3_COMPROMETIDO
        reporta(abs(Z - 969.92) < 0.05,
                f"óptimo de referencia Z* = {Z:.2f} m³/d (esperado 969.92)",
                f"Z* = {Z:.2f} m³/d, distinto del esperado 969.92 — avisa al instructor")
    except Exception as e:  # noqa: BLE001
        reporta(False, "", f"el simulador falló: {type(e).__name__}: {e}")
else:
    print("  (omitido: corrige primero los puntos marcados con [X])")

print()
if errores == 0:
    print("INSTALACIÓN CORRECTA — tu computadora está lista para las 4 sesiones.")
else:
    print(f"Hay {errores} punto(s) por corregir. Revisa la sección «Problemas frecuentes» de la guía.")
