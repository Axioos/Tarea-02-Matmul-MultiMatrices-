# Makefile para matmul.cu

# Compilador CUDA
NVCC = nvcc

# Opciones de compilación
NVCC_FLAGS = -O3 -arch=sm_75 -Xcompiler -fopenmp -lpthread
# Nota: Cambia -arch=sm_75 según tu GPU:
# - sm_50: Maxwell
# - sm_60, sm_61: Pascal
# - sm_70, sm_72: Volta
# - sm_75: Turing
# - sm_80, sm_86: Ampere
# - sm_89: Ada Lovelace
# - sm_90: Hopper

# Nombre del ejecutable
TARGET = prog

# Archivos fuente
SRC = matmul.cu

# Regla por defecto
all: $(TARGET)

# Compilar el programa
$(TARGET): $(SRC)
	@echo "Compilando $(TARGET)..."
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)
	@echo "Compilación completada. Ejecutar con: ./$(TARGET) <n> <nt> <ALG>"

# Regla para compilar con información de debug
debug: NVCC_FLAGS += -G -g
debug: $(TARGET)

# Regla para compilar con comprobación de errores CUDA
check: NVCC_FLAGS += -lineinfo -Xptxas -v
check: $(TARGET)

# Regla para compilar para diferentes arquitecturas
# Para compatibilidad más amplia (puede ser más lento)
compatible: NVCC_FLAGS = -O3 -arch=sm_50 -gencode=arch=compute_50,code=sm_50 \
                         -gencode=arch=compute_60,code=sm_60 \
                         -gencode=arch=compute_70,code=sm_70 \
                         -gencode=arch=compute_75,code=sm_75 \
                         -Xcompiler -fopenmp -lpthread
compatible: $(TARGET)

# Limpiar archivos generados
clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f core*

# Ejemplos de ejecución
test: $(TARGET)
	@echo ""
	@echo "Ejecutando pruebas..."
	@echo "====================="
	@echo "1. CPU con 4 hilos (matriz 512x512):"
	@./$(TARGET) 512 4 1
	@echo ""
	@echo "2. GPU básica (matriz 512x512):"
	@./$(TARGET) 512 1 2
	@echo ""
	@echo "3. GPU con memoria compartida (matriz 512x512):"
	@./$(TARGET) 512 1 3

benchmark: $(TARGET)
	@echo ""
	@echo "Ejecutando benchmark..."
	@echo "======================="
	@echo "Tamaño | Algoritmo | Tiempo (s) | GFLOPs"
	@echo "-------|-----------|------------|--------"
	@for size in 256 512 1024 2048; do \
		for alg in 1 2 3; do \
			if [ $$alg -eq 1 ]; then threads=4; else threads=1; fi; \
			echo -n "$${size}x$${size} | "; \
			if [ $$alg -eq 1 ]; then echo -n "CPU      | "; \
			elif [ $$alg -eq 2 ]; then echo -n "GPU      | "; \
			else echo -n "GPUsm    | "; fi; \
			./$(TARGET) $$size $$threads $$alg 2>/dev/null | \
			grep "Tiempo de cómputo:" | \
			awk '{printf "%.6f | ", $$4}'; \
			./$(TARGET) $$size $$threads $$alg 2>/dev/null | \
			grep "Tasa de cómputo:" | \
			awk '{printf "%.2f\n", $$4}'; \
		done; \
	done

profile: $(TARGET)
	@echo "Ejecutando con profiling (necesita nvprof/ncu)..."
	@echo "CPU (n=1024, 4 hilos):"
	nvprof ./$(TARGET) 1024 4 1
	@echo ""
	@echo "GPU básica (n=1024):"
	nvprof ./$(TARGET) 1024 1 2
	@echo ""
	@echo "GPU con memoria compartida (n=1024):"
	nvprof ./$(TARGET) 1024 1 3

# Ayuda
help:
	@echo "Makefile para compilar matmul.cu"
	@echo ""
	@echo "Targets disponibles:"
	@echo "  all        : Compila el programa (por defecto)"
	@echo "  debug      : Compila con información de debug"
	@echo "  check      : Compila con información detallada"
	@echo "  compatible : Compila para múltiples arquitecturas"
	@echo "  clean      : Limpia archivos generados"
	@echo "  test       : Ejecuta pruebas básicas"
	@echo "  benchmark  : Ejecuta benchmark con diferentes tamaños"
	@echo "  profile    : Ejecuta con profiling (necesita nvprof)"
	@echo "  help       : Muestra esta ayuda"
	@echo ""
	@echo "Uso del programa:"
	@echo "  ./prog <n> <nt> <ALG>"
	@echo "    n   : tamaño de la matriz (n x n)"
	@echo "    nt  : número de hilos CPU (solo para ALG=1)"
	@echo "    ALG : algoritmo (1=CPU, 2=GPU, 3=GPUsm)"
	@echo ""
	@echo "Ejemplos:"
	@echo "  ./prog 1024 4 1   # CPU con 4 hilos"
	@echo "  ./prog 1024 1 2   # GPU básica"
	@echo "  ./prog 1024 1 3   # GPU con memoria compartida"

.PHONY: all clean test benchmark profile help debug check compatible
