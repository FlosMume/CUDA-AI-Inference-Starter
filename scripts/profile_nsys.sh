#!/usr/bin/env bash
set -euo pipefail
if ! command -v nsys &> /dev/null; then
  echo "Nsight Systems (nsys) not found in PATH."
  exit 1
fi
out=cuda_ai_infer
nsys profile -t cuda,osrt,mpi,nvtx -o ${out} "$@"
echo "Generated ${out}.qdrep (open in Nsight Systems GUI)."
