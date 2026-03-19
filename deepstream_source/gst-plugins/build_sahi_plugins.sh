#!/usr/bin/env bash
#
# Build only the SAHI GStreamer plugins (preprocess + postprocess).
# Run from inside /opt/nvidia/deepstream/deepstream/sources/gst-plugins/
# after copying the plugin directories there.
#
set -euo pipefail

if [[ -z "${CUDA_VER:-}" ]]; then
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    else
        echo "ERROR: CUDA_VER not set and nvcc not found." >&2
        exit 1
    fi
fi
export CUDA_VER

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for plugin in gst-nvsahipreprocess gst-nvsahipostprocess; do
    echo "=== Building ${plugin} (CUDA ${CUDA_VER}) ==="
    make -C "${SCRIPT_DIR}/${plugin}" clean
    make -C "${SCRIPT_DIR}/${plugin}" -j"$(nproc)" CUDA_VER="${CUDA_VER}"
    make -C "${SCRIPT_DIR}/${plugin}" install CUDA_VER="${CUDA_VER}"
done

echo "=== SAHI plugins installed ==="
