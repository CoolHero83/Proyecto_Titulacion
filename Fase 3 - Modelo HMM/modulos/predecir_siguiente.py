#!/usr/bin/env python3
# encoding: utf-8
"""
Módulo: predecir_siguiente
==========================
Dada una secuencia parcial de fases de un autor, predice la siguiente
fase más probable de la Cyber Kill Chain usando el modelo HMM entrenado.

Funciones exportadas:
  - predecir_siguiente_fase(modelo, secuencia, mapeo) -> dict
  - analizar_autores(modelo, secuencias_por_autor, mapeo) -> dict
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def predecir_siguiente_fase(modelo, secuencia: List[int], mapeo: dict) -> dict:
    """
    Predice la siguiente fase de la Cyber Kill Chain más probable
    dada una secuencia parcial de observaciones.

    Usa el algoritmo Forward para calcular la probabilidad del siguiente
    estado oculto y de ahí la observación más probable.

    Parámetros
    ----------
    modelo : CategoricalHMM
        Modelo HMM entrenado.
    secuencia : list[int]
        Lista de fases observadas hasta ahora (IDs 0-5).
    mapeo : dict
        Diccionario con el mapeo Kill Chain (para nombres de fases).

    Retorna
    -------
    dict
        Diccionario con:
        - fase_predicha: ID de la fase más probable (0-5)
        - nombre_fase: nombre de la fase predicha
        - probabilidades: dict {nombre_fase: probabilidad}
        - estado_oculto: ID del estado oculto más probable
        - confianza: probabilidad de la predicción
    """
    if not secuencia:
        return {
            "fase_predicha": None,
            "nombre_fase": "Sin datos",
            "probabilidades": {},
            "estado_oculto": None,
            "confianza": 0.0
        }

    # Convertir a array numpy
    X = np.array(secuencia, dtype=int).reshape(-1, 1)

    # Calcular la distribución de estados ocultos para el último paso
    # usando el algoritmo Forward
    log_prob, estado_actual = modelo.decode(X, algorithm="viterbi")
    ultimo_estado = int(estado_actual[-1])

    # Obtener distribución de transición desde el último estado
    transiciones = modelo.transmat_[ultimo_estado]

    # Obtener matriz de emisión
    emisiones = modelo.emissionprob_

    # Calcular probabilidad de cada observación (fase) en el siguiente paso
    n_fases = emisiones.shape[1]
    prob_siguiente_fase = np.zeros(n_fases)

    for s in range(modelo.n_components):
        prob_estado = transiciones[s]
        for f in range(n_fases):
            prob_siguiente_fase[f] += prob_estado * emisiones[s, f]

    # Normalizar
    prob_siguiente_fase /= prob_siguiente_fase.sum()

    # Obtener la fase más probable
    fase_predicha = int(np.argmax(prob_siguiente_fase))
    confianza = float(prob_siguiente_fase[fase_predicha])

    # Obtener nombres de fases
    fases = mapeo.get('metadata', {}).get('fases', [])
    nombre_fase_predicha = fases[fase_predicha]['nombre'] if fase_predicha < len(fases) else "Desconocida"

    # Construir dict de probabilidades por nombre
    prob_por_nombre = {}
    for f in range(n_fases):
        nombre = fases[f]['nombre'] if f < len(fases) else f"Fase {f}"
        prob_por_nombre[nombre] = float(prob_siguiente_fase[f])

    return {
        "fase_predicha": fase_predicha,
        "nombre_fase": nombre_fase_predicha,
        "probabilidades": prob_por_nombre,
        "estado_oculto": int(ultimo_estado),
        "confianza": confianza
    }


def analizar_autores(modelo, secuencias_por_autor: Dict[str, List[int]], mapeo: dict) -> Dict[str, dict]:
    """
    Analiza todos los autores: para cada uno, predice la siguiente fase
    basándose en su historial completo.

    Parámetros
    ----------
    modelo : CategoricalHMM
        Modelo HMM entrenado.
    secuencias_por_autor : dict
        Diccionario {nombre_usuario: [fase_0, ..., fase_n]}.
    mapeo : dict
        Diccionario con el mapeo Kill Chain.

    Retorna
    -------
    dict
        Diccionario {nombre_usuario: dict_con_prediccion}
        Cada dict contiene:
        - secuencia: lista de fases observadas
        - siguiente: resultado de predecir_siguiente_fase
        - estado_dominante: el estado oculto más frecuente
    """
    resultados = {}

    for usuario, secuencia in secuencias_por_autor.items():
        if len(secuencia) < 2:
            # Con 1 solo post no se puede predecir mucho
            resultados[usuario] = {
                "secuencia": secuencia,
                "siguiente": {
                    "fase_predicha": None,
                    "nombre_fase": "Secuencia demasiado corta",
                    "probabilidades": {},
                    "estado_oculto": None,
                    "confianza": 0.0
                },
                "estado_dominante": None,
                "mensaje": "Secuencia demasiado corta (menos de 2 posts)"
            }
            continue

        # Predecir siguiente fase
        prediccion = predecir_siguiente_fase(modelo, secuencia, mapeo)

        # Calcular estado oculto dominante con Viterbi
        X = np.array(secuencia, dtype=int).reshape(-1, 1)
        _, estados = modelo.decode(X, algorithm="viterbi")
        estados_list = [int(e) for e in estados]

        # Estado más frecuente
        from collections import Counter
        estado_dominante = Counter(estados_list).most_common(1)[0][0]

        resultados[usuario] = {
            "secuencia": secuencia,
            "secuencia_estados": estados_list,
            "siguiente": prediccion,
            "estado_dominante": estado_dominante,
            "mensaje": "OK"
        }

    return resultados


def generar_reporte_csv(resultados: Dict[str, dict], mapeo: dict, ruta_salida: str) -> None:
    """
    Genera un archivo CSV con el reporte de análisis por autor.

    Parámetros
    ----------
    resultados : dict
        Resultado de analizar_autores().
    mapeo : dict
        Diccionario con el mapeo Kill Chain.
    ruta_salida : str
        Ruta donde guardar el CSV.
    """
    import pandas as pd
    import os

    fases = mapeo.get('metadata', {}).get('fases', [])
    registros = []

    for usuario, data in resultados.items():
        siguiente = data.get('siguiente', {})
        registros.append({
            "usuario": usuario,
            "cantidad_posts": len(data['secuencia']),
            "estado_dominante": data.get('estado_dominante', ''),
            "siguiente_fase_id": siguiente.get('fase_predicha', ''),
            "siguiente_fase_nombre": siguiente.get('nombre_fase', ''),
            "confianza_prediccion": round(siguiente.get('confianza', 0), 4),
            "mensaje": data.get('mensaje', 'OK')
        })

    df = pd.DataFrame(registros)
    os.makedirs(os.path.dirname(ruta_salida) or '.', exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    logger.info(f"Reporte CSV generado: {ruta_salida} ({len(registros)} autores)")