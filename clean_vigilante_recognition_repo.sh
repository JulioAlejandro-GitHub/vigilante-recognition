#!/usr/bin/env bash
set -euo pipefail

# Limpieza segura para el repo vigilante-recognition
# - Por defecto hace DRY RUN
# - Solo borra artefactos locales comunes
# - NO toca .git ni archivos fuente
#
# Uso:
#   bash clean_vigilante_recognition_repo.sh
#   bash clean_vigilante_recognition_repo.sh --apply
#   bash clean_vigilante_recognition_repo.sh --apply "/ruta/al/repo"

REPO_PATH="${2:-/Users/julio/Desktop/Archivo/Vigilante SW/GIT/vigilante-recognition}"
MODE="${1:-}"

if [[ "$MODE" != "" && "$MODE" != "--apply" ]]; then
  echo "Uso:"
  echo "  bash clean_vigilante_recognition_repo.sh"
  echo "  bash clean_vigilante_recognition_repo.sh --apply"
  echo "  bash clean_vigilante_recognition_repo.sh --apply \"/ruta/al/repo\""
  exit 1
fi

if [[ ! -d "$REPO_PATH" ]]; then
  echo "No existe el directorio: $REPO_PATH" >&2
  exit 1
fi

cd "$REPO_PATH"

echo "==> Repo: $REPO_PATH"
echo "==> Modo: ${MODE:---dry-run}"

TARGETS=(
  ".venv"
  ".pytest_cache"
  ".mypy_cache"
  ".ruff_cache"
  "htmlcov"
  "dist"
  "build"
  ".coverage"
  ".DS_Store"
  ".env"
)

echo
echo "==> Objetivos directos"
for item in "${TARGETS[@]}"; do
  if [[ -e "$item" ]]; then
    echo "  - $item"
  fi
done

echo
echo "==> Objetivos recursivos"
find . -type d \( -name "__pycache__" -o -name ".ipynb_checkpoints" \) -print || true
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".DS_Store" \) -print || true

echo
echo "==> Git status antes de limpiar"
git status --short || true

if [[ "$MODE" != "--apply" ]]; then
  echo
  echo "DRY RUN terminado. No se eliminó nada."
  echo "Ejecuta con --apply para aplicar la limpieza."
  exit 0
fi

echo
echo "==> Aplicando limpieza"

for item in "${TARGETS[@]}"; do
  if [[ -e "$item" ]]; then
    rm -rf "$item"
    echo "Eliminado: $item"
  fi
done

find . -type d \( -name "__pycache__" -o -name ".ipynb_checkpoints" \) -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".DS_Store" \) -delete

echo
echo "==> Git status después de limpiar"
git status --short || true

echo
echo "Limpieza completada."
