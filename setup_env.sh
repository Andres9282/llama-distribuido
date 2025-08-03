#!/bin/bash

'''Ejecutar usando este comando:
./setup_env.sh
'''

# Ruta al entorno virtual
VENV_DIR="llama-env"  # cambia si tu entorno virtual se llama distinto

# Token real de Hugging Face (reemplaza esto)
HF_TOKEN_VALUE="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"

# Archivo de activación del entorno virtual
ACTIVATE_FILE="$VENV_DIR/bin/activate"

# Línea que exporta el token
EXPORT_LINE="export HF_TOKEN=$HF_TOKEN_VALUE"

# Verificar si ya existe
if grep -q "HF_TOKEN" "$ACTIVATE_FILE"; then
    echo "⚠️  La variable HF_TOKEN ya está definida en $ACTIVATE_FILE"
else
    echo "$EXPORT_LINE" >> "$ACTIVATE_FILE"
    echo "✅ Token agregado al entorno virtual en: $ACTIVATE_FILE"
fi
