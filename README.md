# Tarea-02-Matmul-MultiMatrices

para compilar usar: make

para ejecutar usar:

**CPU con 4 hilos, matriz 1024x1024**
./prog 1024 4 1

**GPU básica, matriz 2048x2048**
./prog 2048 1 2

**GPU con memoria compartida, matriz 2048x2048**
./prog 2048 1 3

**Generar resultados**

export LC_ALL=C

make benchmark > resultados.txt

**Generar Graficos**

*REQUIERE matplotlib*
python3 graficos.py
