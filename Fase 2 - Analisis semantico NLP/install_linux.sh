#!/bin/bash
# ============================================================================
# Script de instalación para Linux (Kali/Ubuntu/Debian)
# ============================================================================
# Soluciona errores comunes:
#   1. tokenizers no compatible con Python 3.13
#   2. No space left on device (PyTorch CUDA ~2GB)
#
# Usa PyTorch CPU-only (~200MB en lugar de ~2GB)
# ============================================================================

set -e  # Detener en caso de error

echo "======================================"
echo "  Instalando dependencias - Fase 2"
echo "  Procesamiento NLP con SecureBERT"
echo "======================================"
echo ""

# 1. Verificar Python
echo "[1/5] Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python no encontrado"
    exit 1
fi
$PYTHON --version

# 2. Actualizar pip
echo "[2/5] Actualizando pip..."
$PYTHON -m pip install --upgrade pip

# 3. Liberar espacio en caché de pip
echo "[3/5] Liberando caché de pip..."
$PYTHON -m pip cache purge

# 4. Instalar dependencias (PyTorch CPU para ahorrar espacio)
echo "[4/5] Instalando dependencias del proyecto..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Instalar PyTorch CPU (mucho más pequeño que CUDA ~200MB vs ~2GB)
echo "  → Instalando PyTorch CPU..."
$PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Instalar las demás dependencias
echo "  → Instalando transformers y otras librerías..."
$PYTHON -m pip install transformers pandas numpy scikit-learn tqdm requests jsonlines python-dateutil

# 5. Verificar instalación
echo "[5/5] Verificando instalación..."
$PYTHON -c "
import transformers
import torch
import pandas
import numpy
import sklearn
print(f'✅ transformers: {transformers.__version__}')
print(f'✅ torch: {torch.__version__}')
print(f'✅ pandas: {pandas.__version__}')
print(f'✅ numpy: {numpy.__version__}')
print(f'✅ scikit-learn: {sklearn.__version__}')
print()
if torch.cuda.is_available():
    print('✅ CUDA disponible')
else:
    print('ℹ️  Modo CPU (ahorra ~1.8GB de espacio en disco)')
print()
print('Dependencias instaladas correctamente')
"

echo ""
echo "======================================"
echo "  Instalación completada"
echo "======================================"
echo ""
echo "Para ejecutar el pipeline:"
echo "  cd modulos && $PYTHON main.py --input ../test_input.csv --output-csv ../salida.csv --output-hmm ../secuencias.json"
echo ""
echo "Para liberar aún más espacio:"
echo "  $PYTHON -m pip cache purge"
echo ""