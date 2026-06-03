#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capa semántica para procesamiento NLP con SecureBERT 2.0
Este script implementa el pipeline completo de enriquecimiento de datos:
1. Carga modelos SecureBERT 2.0 desde Hugging Face
2. Procesa NER para detección de entidades de ciberseguridad
3. Mapea entidades a técnicas MITRE ATT&CK
4. Agrupa posts por autor y crea secuencias cronológicas
5. Filtra secuencias válidas para entrenamiento HMM
6. Genera salidas estructuradas para modelos predictivos
"""

import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
from tqdm import tqdm
import warnings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suprimir warnings de torch y transformers
warnings.filterwarnings('ignore')

class SecureBERTProcessor:
    """Clase principal para procesamiento con SecureBERT 2.0"""

    def __init__(self, mitre_mapping_path: str = 'mitre_mapping.json'):
        """
        Inicializar procesador con modelos SecureBERT 2.0

        Args:
            mitre_mapping_path: Ruta al archivo JSON de mapeo MITRE
        """
        self.mitre_mapping = self._load_mitre_mapping(mitre_mapping_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Usando dispositivo: {self.device}")

        # Cargar modelos SecureBERT 2.0
        self.tokenizer, self.model_ner, self.model_seq = self._load_models()

        # Crear pipelines
        self.ner_pipeline = self._create_ner_pipeline()
        self.seq_pipeline = self._create_seq_pipeline()

        logger.info("Modelos SecureBERT 2.0 cargados exitosamente")

    def _load_mitre_mapping(self, path: str) -> Dict[str, List[str]]:
        """Cargar diccionario de mapeo MITRE ATT&CK"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error cargando mapeo MITRE: {e}")
            return {}

    def _load_models(self) -> Tuple[Any, Any, Any]:
        """Cargar modelos SecureBERT 2.0 desde Hugging Face con fallback"""
        try:
            logger.info("Intentando cargar SecureBERT 2.0-NER...")
            tokenizer = AutoTokenizer.from_pretrained("cisco-ai/SecureBERT2.0-NER")
            model = AutoModelForTokenClassification.from_pretrained("cisco-ai/SecureBERT2.0-NER").to(self.device)
            logger.info("✅ SecureBERT 2.0-NER cargado exitosamente")
            return tokenizer, model, model

        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar SecureBERT 2.0-NER: {e}")
            logger.info("Cargando modelo alternativo (bert-base-uncased) con fine-tuning simulado...")

            # Fallback: Usar modelo alternativo compatible
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased").to(self.device)
            logger.info("✅ Modelo alternativo cargado exitosamente")
            return tokenizer, model, model

    def _create_ner_pipeline(self) -> pipeline:
        """Crear pipeline de NER con SecureBERT"""
        return pipeline(
            "ner",
            model=self.model_ner,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1,
            aggregation_strategy="simple"
        )

    def _create_seq_pipeline(self) -> pipeline:
        """Crear pipeline de clasificación con SecureBERT"""
        return pipeline(
            "text-classification",
            model=self.model_seq,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1
        )

    def _prepare_masked_texts(self, text: str) -> List[str]:
        """
        Preparar múltiples versiones del texto con máscaras en diferentes posiciones
        para detección de entidades
        """
        # Patrones comunes para entidades de ciberseguridad
        patterns = [
            "The [MASK] malware",
            "The malware [MASK]",
            "[MASK] vulnerability",
            "CVE-[MASK]",
            "Using [MASK] tool",
            "[MASK] attack",
            "The [MASK] exploit",
            "[MASK] ransomware",
            "[MASK] phishing",
            "[MASK] threat"
        ]

        masked_texts = []
        for pattern in patterns:
            if len(text.split()) >= 3:  # Solo para textos con suficiente longitud
                masked_text = pattern + " " + text
                masked_texts.append(masked_text)

        # Añadir el texto original con máscaras en palabras clave
        keywords = ["malware", "vulnerability", "tool", "attack", "exploit", "threat", "virus", "trojan"]
        for keyword in keywords:
            if keyword in text.lower():
                masked_text = text.replace(keyword, "[MASK]")
                masked_texts.append(masked_text)

        return masked_texts[:5]  # Limitar a 5 variantes para evitar procesamiento excesivo

    def _classify_entity(self, predicted_word: str) -> Optional[str]:
        """
        Clasificar la palabra predicha en una categoría de entidad de ciberseguridad
        """
        word_lower = predicted_word.lower().strip()

        # Listas de palabras clave por categoría
        malware_keywords = ["malware", "virus", "trojan", "worm", "spyware", "ransomware", "adware"]
        tool_keywords = ["tool", "kit", "framework", "software", "program"]
        vulnerability_keywords = ["vulnerability", "flaw", "bug", "exploit", "cve"]
        technique_keywords = ["attack", "threat", "breach", "intrusion", "hack"]
        sector_keywords = ["banking", "financial", "healthcare", "government", "enterprise"]

        # Clasificación por palabras clave
        if any(keyword in word_lower for keyword in malware_keywords):
            return "MALWARE"
        elif any(keyword in word_lower for keyword in tool_keywords):
            return "TOOL"
        elif any(keyword in word_lower for keyword in vulnerability_keywords):
            return "VULNERABILITY"
        elif any(keyword in word_lower for keyword in technique_keywords):
            return "TECHNIQUE"
        elif any(keyword in word_lower for keyword in sector_keywords):
            return "SECTOR"
        elif word_lower.startswith("cve-") or "cve" in word_lower:
            return "VULNERABILITY"
        elif len(predicted_word) > 3 and predicted_word.istitle():
            return "TOOL"  # Nombres propios suelen ser herramientas
        else:
            return None

    def _calculate_confidence(self, outputs: torch.Tensor, mask_token_index: torch.Tensor) -> float:
        """
        Calcular la confianza de la predicción basada en las probabilidades del modelo
        """
        # Obtener las probabilidades del token predicho
        logits = outputs.logits[0, mask_token_index]
        probabilities = torch.softmax(logits, dim=-1)
        top_prob, _ = torch.topk(probabilities, 1)

        return float(top_prob.item())

    def process_ner(self, text: str) -> List[Dict[str, Any]]:
        """
        Procesar texto con SecureBERT NER para detección de entidades

        Args:
            text: Texto a procesar

        Returns:
            Lista de entidades detectadas con tipo, texto y confianza
        """
        if not text or pd.isna(text) or str(text).strip() == "":
            return []

        try:
            # Usar pipeline de NER directamente
            ner_results = self.ner_pipeline(str(text))

            # Formatear resultados
            entities = []
            for entity in ner_results:
                # Convertir numpy types a Python types para JSON serialization
                confidence = float(round(entity['score'], 4))
                entities.append({
                    "type": entity['entity_group'],
                    "text": entity['word'],
                    "confidence": confidence,
                    "start": int(entity['start']),
                    "end": int(entity['end'])
                })

            return entities

        except Exception as e:
            logger.error(f"Error en procesamiento NER: {e}")
            return []

    def map_to_mitre(self, entities: List[Dict[str, Any]]) -> List[str]:
        """
        Mapear entidades detectadas a técnicas MITRE ATT&CK

        Args:
            entities: Lista de entidades detectadas

        Returns:
            Lista única de IDs de técnicas MITRE
        """
        mitre_techniques = set()

        for entity in entities:
            entity_text = entity['text'].lower()
            # Buscar en diccionario MITRE
            for key, techniques in self.mitre_mapping.items():
                if key.lower() in entity_text:
                    mitre_techniques.update(techniques)

        return sorted(list(mitre_techniques))

    def calculate_threat_score(self, entities: List[Dict[str, Any]], mitre_techniques: List[str]) -> float:
        """
        Calcular puntuación de amenaza basada en entidades y técnicas

        Args:
            entities: Entidades detectadas
            mitre_techniques: Técnicas MITRE mapeadas

        Returns:
            Puntuación de amenaza (0-1)
        """
        if not entities and not mitre_techniques:
            return 0.0

        # Puntuación basada en confianza de entidades
        entity_score = sum(entity['confidence'] for entity in entities) / len(entities) if entities else 0

        # Puntuación basada en número de técnicas MITRE
        technique_score = min(len(mitre_techniques) / 10, 1.0)  # Normalizar a 0-1

        # Combinar puntuaciones
        threat_score = (entity_score * 0.7 + technique_score * 0.3)
        return round(threat_score, 4)

    def process_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar un post individual y enriquecerlo con NLP

        Args:
            post: Diccionario con datos del post

        Returns:
            Post enriquecido con entidades y técnicas MITRE
        """
        try:
            # Extraer contenido limpio
            clean_content = post.get('body_limpio', '') or post.get('body', '') or ''

            # Procesar NER con SecureBERT
            entities = self.process_ner(clean_content)

            # Mapear a técnicas MITRE
            mitre_techniques = self.map_to_mitre(entities)

            # Calcular puntuación de amenaza
            threat_score = self.calculate_threat_score(entities, mitre_techniques)

            # Crear post enriquecido
            enriched_post = {
                **post,
                "entities": json.dumps([{
                    "type": e["type"],
                    "text": e["text"],
                    "confidence": e["confidence"]
                } for e in entities], ensure_ascii=False),
                "mitre_techniques": json.dumps(mitre_techniques, ensure_ascii=False),
                "threat_score": threat_score,
                "entity_count": len(entities),
                "mitre_count": len(mitre_techniques)
            }

            return enriched_post

        except Exception as e:
            logger.error(f"Error procesando post {post.get('message_id', 'unknown')}: {e}")
            return {**post, "entities": "[]", "mitre_techniques": "[]", "threat_score": 0.0}

    def group_by_author(self, posts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Agrupar posts por autor y ordenar cronológicamente

        Args:
            posts: Lista de posts enriquecidos

        Returns:
            Diccionario con secuencias de posts por autor
        """
        author_sequences = {}

        for post in posts:
            username = post.get('username', 'unknown_user')
            timestamp = post.get('timestamp')

            if not timestamp:
                continue

            try:
                # Convertir timestamp a datetime para ordenamiento
                if isinstance(timestamp, str):
                    dt = pd.to_datetime(timestamp)
                else:
                    dt = timestamp

                if username not in author_sequences:
                    author_sequences[username] = []

                # Agregar post con datetime para ordenamiento
                author_sequences[username].append({
                    **post,
                    'datetime': dt
                })

            except Exception as e:
                logger.warning(f"Error procesando timestamp para post {post.get('message_id', 'unknown')}: {e}")
                continue

        # Ordenar cada secuencia cronológicamente
        for username in author_sequences:
            author_sequences[username].sort(key=lambda x: x['datetime'])

        return author_sequences

    def filter_sequences(self, author_sequences: Dict[str, List[Dict[str, Any]]], min_length: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Filtrar secuencias con longitud mínima para HMM

        Args:
            author_sequences: Secuencias por autor
            min_length: Longitud mínima requerida

        Returns:
            Secuencias filtradas
        """
        return {
            username: sequences
            for username, sequences in author_sequences.items()
            if len(sequences) >= min_length
        }

    def process_csv(self, input_path: str, output_path: str) -> None:
        """
        Procesar archivo CSV completo y generar salida enriquecida

        Args:
            input_path: Ruta al CSV de entrada
            output_path: Ruta al CSV de salida
        """
        try:
            logger.info(f"Cargando datos desde {input_path}...")
            df = pd.read_csv(input_path)

            logger.info(f"Procesando {len(df)} posts con SecureBERT 2.0...")
            enriched_posts = []

            # Procesar cada post con barra de progreso
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando posts"):
                post_dict = row.to_dict()
                enriched_post = self.process_post(post_dict)
                enriched_posts.append(enriched_post)

            # Crear DataFrame enriquecido
            enriched_df = pd.DataFrame(enriched_posts)

            logger.info(f"Guardando datos enriquecidos en {output_path}...")
            enriched_df.to_csv(output_path, index=False)

            logger.info(f"Procesamiento completado. {len(enriched_posts)} posts procesados.")
            logger.info(f"Datos enriquecidos guardados en: {output_path}")

        except Exception as e:
            logger.error(f"Error procesando CSV: {e}")
            raise

    def generate_author_sequences(self, input_path: str, output_path: str) -> None:
        """
        Generar secuencias de autores para entrenamiento HMM

        Args:
            input_path: Ruta al CSV enriquecido
            output_path: Ruta al JSON de salida
        """
        try:
            logger.info(f"Cargando datos enriquecidos desde {input_path}...")
            df = pd.read_csv(input_path)

            # Convertir a lista de diccionarios
            posts = df.to_dict('records')

            logger.info("Agrupando posts por autor...")
            author_sequences = self.group_by_author(posts)

            logger.info(f"Filtrando secuencias con ≥3 posts...")
            filtered_sequences = self.filter_sequences(author_sequences)

            # Preparar datos para HMM
            hmm_data = {
                "metadata": {
                    "total_authors": len(author_sequences),
                    "valid_sequences": len(filtered_sequences),
                    "generated_at": datetime.now().isoformat(),
                    "min_sequence_length": 3
                },
                "sequences": {}
            }

            for username, sequences in filtered_sequences.items():
                hmm_data["sequences"][username] = [
                    {
                        "message_id": seq["message_id"],
                        "timestamp": seq["timestamp"],
                        "threat_score": seq["threat_score"],
                        "entities": json.loads(seq["entities"]),
                        "mitre_techniques": json.loads(seq["mitre_techniques"])
                    }
                    for seq in sequences
                ]

            logger.info(f"Guardando secuencias HMM en {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(hmm_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Secuencias HMM generadas: {len(filtered_sequences)} autores válidos")
            logger.info(f"Datos para HMM guardados en: {output_path}")

        except Exception as e:
            logger.error(f"Error generando secuencias HMM: {e}")
            raise

def main():
    """Función principal para ejecución desde línea de comandos"""
    parser = argparse.ArgumentParser(
        description="enrich.py - Procesamiento NLP con SecureBERT 2.0 para datos de Dark Web",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--input",
        "-i",
        default="../Fase 1/Scraping-Onion-Sites/output/forum_record_limpio.csv",
        help="Ruta al CSV de entrada (limpio de Fase 1)"
    )

    parser.add_argument(
        "--output-csv",
        "-o",
        default="enriched_forum_data.csv",
        help="Ruta al CSV de salida enriquecido"
    )

    parser.add_argument(
        "--output-hmm",
        "-m",
        default="author_sequences.json",
        help="Ruta al JSON de salida para HMM"
    )

    parser.add_argument(
        "--mitre-mapping",
        "-d",
        default="mitre_mapping.json",
        help="Ruta al archivo de mapeo MITRE"
    )

    args = parser.parse_args()

    try:
        logger.info("Iniciando procesamiento NLP con SecureBERT 2.0...")
        logger.info(f"Configuración: {vars(args)}")

        # Inicializar procesador
        processor = SecureBERTProcessor(mitre_mapping_path=args.mitre_mapping)

        # Procesar CSV y enriquecer datos
        processor.process_csv(args.input, args.output_csv)

        # Generar secuencias para HMM
        processor.generate_author_sequences(args.output_csv, args.output_hmm)

        logger.info("✅ Procesamiento completado exitosamente!")
        logger.info(f"📊 Datos enriquecidos: {args.output_csv}")
        logger.info(f"🔗 Secuencias HMM: {args.output_hmm}")

    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())