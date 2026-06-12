#!/usr/bin/env python3
# encoding: utf-8
"""
Módulo: visualizar_resultados
=============================
Genera gráficos y visualizaciones del modelo HMM y los resultados
del análisis de comportamiento de autores.

Funciones exportadas:
  - graficar_matriz_transicion(modelo, mapeo, ruta)
  - graficar_distribucion_estados(resultados, mapeo, ruta)
  - graficar_predicciones_autor(resultados, mapeo, ruta)
  - generar_dashboard_completo(modelo, resultados, mapeo, directorio)
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Para entornos sin display
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Paleta de colores para las fases de Kill Chain
COLORES_FASES = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']


def graficar_matriz_transicion(modelo, mapeo: dict, ruta: str) -> None:
    """
    Genera un heatmap de la matriz de transición entre estados ocultos.

    Parámetros
    ----------
    modelo : CategoricalHMM
        Modelo HMM entrenado.
    mapeo : dict
        Diccionario con el mapeo Kill Chain.
    ruta : str
        Ruta donde guardar la imagen PNG.
    """
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)

    transmat = modelo.transmat_
    n_estados = transmat.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(transmat, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'Estado {i}' for i in range(n_estados)],
                yticklabels=[f'Estado {i}' for i in range(n_estados)],
                ax=ax, vmin=0, vmax=1)

    ax.set_title('Matriz de Transición entre Estados Ocultos', fontsize=14, fontweight='bold')
    ax.set_xlabel('Estado Siguiente', fontsize=12)
    ax.set_ylabel('Estado Actual', fontsize=12)

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Matriz de transición guardada en: {ruta}")


def graficar_distribucion_estados(resultados: Dict[str, dict], mapeo: dict, ruta: str) -> None:
    """
    Genera un gráfico de barras con la distribución de estados dominantes
    entre los autores analizados.

    Parámetros
    ----------
    resultados : dict
        Resultado de analizar_autores().
    mapeo : dict
        Diccionario con el mapeo Kill Chain.
    ruta : str
        Ruta donde guardar la imagen PNG.
    """
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)

    # Contar estados dominantes
    from collections import Counter
    estados = []
    for usuario, data in resultados.items():
        ed = data.get('estado_dominante')
        if ed is not None:
            estados.append(ed)

    if not estados:
        logger.warning("No hay datos de estados para graficar")
        return

    conteo = Counter(estados)
    estados_ordenados = sorted(conteo.keys())
    valores = [conteo[e] for e in estados_ordenados]
    etiquetas = [f'Estado {e}' for e in estados_ordenados]

    fig, ax = plt.subplots(figsize=(10, 6))
    colores = [COLORES_FASES[e % len(COLORES_FASES)] for e in estados_ordenados]
    ax.bar(etiquetas, valores, color=colores, edgecolor='black', linewidth=0.5)

    ax.set_title('Distribución de Estados Ocultos entre Autores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Estado Oculto (Perfil de Comportamiento)', fontsize=12)
    ax.set_ylabel('Cantidad de Autores', fontsize=12)

    # Agregar valores sobre las barras
    for i, v in enumerate(valores):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Distribución de estados guardada en: {ruta}")


def graficar_secuencia_autor(usuario: str, data: dict, mapeo: dict, ruta: str) -> None:
    """
    Genera un gráfico de línea de tiempo para un autor específico,
    mostrando la secuencia de fases y las predicciones.

    Parámetros
    ----------
    usuario : str
        Nombre del usuario.
    data : dict
        Datos del autor (de analizar_autores).
    mapeo : dict
        Diccionario con el mapeo Kill Chain.
    ruta : str
        Ruta donde guardar la imagen PNG.
    """
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)

    fases = mapeo.get('metadata', {}).get('fases', [])
    secuencia = data['secuencia']
    siguiente = data.get('siguiente', {})
    prediccion = siguiente.get('fase_predicha')

    # Determinar el rango de fases
    max_fase = max(max(secuencia), prediccion if prediccion is not None else 0)

    fig, ax = plt.subplots(figsize=(max(8, len(secuencia) * 1.2), 5))

    # Graficar secuencia observada
    x = list(range(len(secuencia)))
    colores_seq = [COLORES_FASES[f % len(COLORES_FASES)] for f in secuencia]
    ax.scatter(x, secuencia, c=colores_seq, s=150, zorder=5, edgecolors='black', linewidth=0.5)

    # Conectar puntos con línea
    ax.plot(x, secuencia, 'b-', alpha=0.4, linewidth=2)

    # Marcar predicción
    if prediccion is not None:
        x_pred = len(secuencia)
        color_pred = COLORES_FASES[prediccion % len(COLORES_FASES)]
        ax.scatter(x_pred, prediccion, marker='*', s=300, c=color_pred,
                   edgecolors='black', linewidth=1, zorder=6, label='Predicción')
        ax.annotate(f'→ {fases[prediccion]["nombre"]}',
                    xy=(x_pred, prediccion),
                    xytext=(x_pred + 0.3, prediccion + 0.3),
                    fontsize=9, fontweight='bold')

    # Configurar ejes
    ax.set_xlabel('Posts (orden cronológico)', fontsize=12)
    ax.set_ylabel('Fase de Kill Chain', fontsize=12)
    ax.set_title(f'Evolución de {usuario}', fontsize=14, fontweight='bold')
    ax.set_yticks(range(max_fase + 1))
    ax.set_yticklabels([f['nombre'] for f in fases[:max_fase + 1]], fontsize=8)
    ax.set_xlim(-0.5, len(secuencia) + 1.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Secuencia de autor guardada en: {ruta}")


def graficar_probabilidades_prediccion(resultados: Dict[str, dict], mapeo: dict, ruta: str) -> None:
    """
    Genera un gráfico de barras apiladas mostrando las probabilidades
    de cada posible siguiente fase para todos los autores.

    Parámetros
    ----------
    resultados : dict
        Resultado de analizar_autores().
    mapeo : dict
        Diccionario con el mapeo Kill Chain.
    ruta : str
        Ruta donde guardar la imagen PNG.
    """
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)

    fases = mapeo.get('metadata', {}).get('fases', [])
    n_fases = len(fases)

    # Recolectar confianzas de predicción por fase
    confianzas_por_fase = {f['nombre']: [] for f in fases}

    for usuario, data in resultados.items():
        siguiente = data.get('siguiente', {})
        probs = siguiente.get('probabilidades', {})
        for nombre_fase, prob in probs.items():
            if nombre_fase in confianzas_por_fase:
                confianzas_por_fase[nombre_fase].append(prob)

    # Calcular promedio
    nombres = []
    promedios = []
    for f in fases:
        nombre = f['nombre']
        vals = confianzas_por_fase.get(nombre, [])
        if vals:
            nombres.append(nombre)
            promedios.append(np.mean(vals))

    if not nombres:
        logger.warning("No hay suficientes datos para graficar probabilidades")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colores = [COLORES_FASES[i % len(COLORES_FASES)] for i in range(len(nombres))]
    ax.barh(nombres, promedios, color=colores, edgecolor='black', linewidth=0.5)

    ax.set_title('Probabilidad Promedio de Siguiente Fase (todos los autores)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Probabilidad Promedio', fontsize=12)
    ax.set_ylabel('Fase de Kill Chain', fontsize=12)
    ax.set_xlim(0, 1)

    for i, v in enumerate(promedios):
        ax.text(v + 0.01, i, f'{v:.2%}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Probabilidades de predicción guardadas en: {ruta}")


def generar_dashboard_completo(modelo, resultados: Dict[str, dict], mapeo: dict, directorio: str) -> None:
    """
    Genera todas las visualizaciones en el directorio especificado.

    Parámetros
    ----------
    modelo : CategoricalHMM
        Modelo HMM entrenado.
    resultados : dict
        Resultado de analizar_autores().
    mapeo : dict
        Diccionario con el mapeo Kill Chain.
    directorio : str
        Directorio donde guardar las imágenes.
    """
    os.makedirs(directorio, exist_ok=True)

    logger.info("Generando dashboard de visualizaciones...")

    # 1. Matriz de transición
    graficar_matriz_transicion(modelo, mapeo, os.path.join(directorio, 'matriz_transicion.png'))

    # 2. Distribución de estados
    graficar_distribucion_estados(resultados, mapeo, os.path.join(directorio, 'distribucion_estados.png'))

    # 3. Probabilidades de predicción
    graficar_probabilidades_prediccion(resultados, mapeo, os.path.join(directorio, 'probabilidades_prediccion.png'))

    # 4. Secuencias de los top 5 autores
    from collections import Counter
    estados_autor = {}
    for usuario, data in resultados.items():
        ed = data.get('estado_dominante')
        if ed is not None:
            estados_autor[usuario] = ed

    top_autores = [u for u, _ in Counter(estados_autor).most_common(5)]
    for usuario in top_autores:
        if usuario in resultados:
            ruta_img = os.path.join(directorio, f'secuencia_{usuario.replace("/", "_")}.png')
            graficar_secuencia_autor(usuario, resultados[usuario], mapeo, ruta_img)

    logger.info(f"Dashboard completo generado en: {directorio}")