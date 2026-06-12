#!/usr/bin/env python3
# encoding: utf-8
"""
Módulo: entrenar_hmm
====================
Entrena un modelo oculto de Markov (HMM) con las secuencias de fases
de la Cyber Kill Chain por autor.

Usa hmmlearn con CategoricalHMM para observaciones discretas (Opción A):
cada post se representa como una fase (0-5) de la Kill Chain.

Funciones exportadas:
  - entrenar_hmm(secuencias, n_estados, n_iter, random_state) -> CategoricalHMM
  - guardar_modelo(modelo, ruta) -> None
  - cargar_modelo(ruta) -> CategoricalHMM | None
"""

import logging
import pickle
import os
from typing import Dict, List, Optional

import numpy as np
from hmmlearn import hmm

logger = logging.getLogger(__name__)


def entrenar_hmm(secuencias_por_autor: Dict[str, List[int]],
                 n_estados: int = 4,
                 n_iter: int = 100,
                 random_state: int = 42) -> Optional[hmm.CategoricalHMM]:
    """
    Entrena un modelo HMM categórico usando las secuencias de fases
    de la Cyber Kill Chain por autor.

    Los estados ocultos representan perfiles de comportamiento del atacante.
    Las observaciones son las fases discretas (0-5) de la Kill Chain.

    Parámetros
    ----------
    secuencias_por_autor : dict
        Diccionario {nombre_usuario: [fase_0, fase_1, ..., fase_n]}.
    n_estados : int
        Número de estados ocultos (perfiles de comportamiento).
        Por defecto 4, que es un buen punto de partida.
    n_iter : int
        Número máximo de iteraciones para Baum-Welch.
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    CategoricalHMM or None
        Modelo HMM entrenado, o None si hay muy pocos datos.
    """
    # Validar que haya suficientes datos
    if not secuencias_por_autor:
        logger.error("No hay secuencias para entrenar el HMM")
        return None

    total_posts = sum(len(seq) for seq in secuencias_por_autor.values())
    if total_posts < n_estados * 3:
        logger.warning(f"Pocos datos ({total_posts} posts) para {n_estados} estados")
        return None

    # Determinar el número de fases (observaciones posibles)
    todas_fases = set()
    for seq in secuencias_por_autor.values():
        todas_fases.update(seq)
    n_fases = max(todas_fases) + 1 if todas_fases else 6

    logger.info(f"Entrenando HMM: {n_estados} estados, {n_fases} observaciones, {total_posts} posts")

    # Preparar datos de entrenamiento: concatenar todas las secuencias
    # hmmlearn requiere un array 1D concatenado + longitudes
    X = []
    lengths = []
    for usuario, seq in secuencias_por_autor.items():
        if len(seq) >= 2:  # Mínimo 2 posts para una secuencia útil
            X.extend(seq)
            lengths.append(len(seq))

    if not X:
        logger.error("Ninguna secuencia tiene al menos 2 posts")
        return None

    X = np.array(X, dtype=int).reshape(-1, 1)

    # Inicializar y entrenar el modelo
    modelo = hmm.CategoricalHMM(
        n_components=n_estados,
        random_state=random_state,
        n_iter=n_iter,
        tol=1e-4,
        verbose=False
    )

    try:
        modelo.fit(X, lengths=lengths)
        logger.info(f"HMM entrenado exitosamente en {modelo.monitor_.iter} iteraciones")
        logger.info(f"Log-verosimilitud final: {modelo.monitor_.history[-1]:.2f}")

        # Mostrar matriz de transición
        logger.info("Matriz de transición entre estados:")
        for i, row in enumerate(modelo.transmat_):
            logger.info(f"  Estado {i}: [{', '.join(f'{p:.3f}' for p in row)}]")

        return modelo

    except Exception as e:
        logger.error(f"Error entrenando HMM: {e}")
        return None


def guardar_modelo(modelo: hmm.CategoricalHMM, ruta: str) -> None:
    """
    Guarda el modelo HMM entrenado en un archivo pickle.

    Parámetros
    ----------
    modelo : CategoricalHMM
        Modelo HMM entrenado.
    ruta : str
        Ruta donde guardar el modelo (ej. ../Datos/modelo_hmm.pkl).
    """
    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)
    try:
        with open(ruta, 'wb') as f:
            pickle.dump(modelo, f)
        logger.info(f"Modelo HMM guardado en: {ruta}")
    except Exception as e:
        logger.error(f"Error guardando modelo en {ruta}: {e}")


def cargar_modelo(ruta: str) -> Optional[hmm.CategoricalHMM]:
    """
    Carga un modelo HMM previamente entrenado desde un archivo pickle.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo del modelo.

    Retorna
    -------
    CategoricalHMM or None
        Modelo cargado, o None si hay error.
    """
    if not os.path.exists(ruta):
        logger.warning(f"No se encontró modelo en: {ruta}")
        return None

    try:
        with open(ruta, 'rb') as f:
            modelo = pickle.load(f)
        logger.info(f"Modelo HMM cargado desde: {ruta}")
        return modelo
    except Exception as e:
        logger.error(f"Error cargando modelo desde {ruta}: {e}")
        return None