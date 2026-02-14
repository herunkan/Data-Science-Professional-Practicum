#!/usr/bin/env bash
set -euo pipefail

SIZES=(512 1024 2048)
REPEATS_CPU=1
REPEATS_GPU=10

echo "Implementation,N,Repeats,Time"
for n in "${SIZES[@]}"; do
  ./cpu/matrix_cpu "$n" "$REPEATS_CPU" | rg "^CSV" | awk -F, '{print $2","$3","$4","$5}'
done

for n in "${SIZES[@]}"; do
  ./cuda/matrix_gpu "$n" "$REPEATS_GPU" | rg "^CSV" | awk -F, '{print $2","$3","$4","$5}'
  ./cuda/matrix_tiled "$n" "$REPEATS_GPU" | rg "^CSV" | awk -F, '{print $2","$3","$4","$5}'
  ./cuda/matrix_cublas "$n" "$REPEATS_GPU" | rg "^CSV" | awk -F, '{print $2","$3","$4","$5}'
done
