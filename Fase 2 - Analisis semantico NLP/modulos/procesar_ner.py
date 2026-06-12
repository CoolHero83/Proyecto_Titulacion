#!/usr/bin/env python3
# encoding: utf-8
"""
Módulo: procesar_ner
=====================
Encargado del reconocimiento de entidades nombradas (NER) usando SecureBERT.
Procesa textos con el pipeline NER de Hugging Face, filtra por confianza
y deduplica resultados.

Funciones exportadas:
  - procesar_ner(texto, pipeline_ner) -> list[dict]
"""

import logging
from typing import List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Umbral mínimo de confianza para considerar una entidad válida
UMBRAL_CONFIANZA = 0.5


def procesar_ner(texto: str, pipeline_ner) -> List[Dict[str, Any]]:
    """
    Procesa un texto con el pipeline NER de SecureBERT para detectar
    entidades de ciberseguridad, filtra por confianza y deduplica resultados.

    Parámetros
    ----------
    texto : str
        Texto a analizar.
    pipeline_ner : pipeline
        Pipeline de Hugging Face configurado para NER.

    Retorna
    -------
    list[dict]
        Lista de entidades detectadas, filtradas y deduplicadas.
        Cada entidad tiene:
        - type: categoría de la entidad (ej. MALWARE, TOOL, VULNERABILITY)
        - text: texto extraído
        - confidence: confianza de la detección (0.0 a 1.0)
        - start: posición inicial en el texto
        - end: posición final en el texto
        Si el texto está vacío o hay error, retorna lista vacía.
    """
    # Validar entrada
    if not texto or pd.isna(texto) or str(texto).strip() == "":
        return []

    try:
        resultados = pipeline_ner(str(texto))

        entidades = []
        for entidad in resultados:
            confianza = float(round(entidad['score'], 4))

            # Filtrar por umbral de confianza mínimo
            if confianza < UMBRAL_CONFIANZA:
                continue

            entidades.append({
                "type": entidad['entity_group'],
                "text": entidad['word'],
                "confidence": confianza,
                "start": int(entidad['start']),
                "end": int(entidad['end'])
            })

        # Deduplicar: misma entidad (type + text) solo una vez
        vistas = set()
        entidades_unicas = []
        for entidad in entidades:
            clave = (entidad['type'], entidad['text'].lower())
            if clave not in vistas:
                vistas.add(clave)
                entidades_unicas.append(entidad)

        return entidades_unicas

    except Exception as e:
        logger.error(f"Error en procesamiento NER: {e}")
        return []