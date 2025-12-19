NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_75 -Xcompiler -fopenmp -lpthread
TARGET = prog
SRC = matmul.cu

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "Compilando $(TARGET)..."
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)
	@echo "Compilación completada. Ejecutar con: ./$(TARGET) <n> <nt> <ALG>"

debug: NVCC_FLAGS += -G -g
debug: $(TARGET)

check: NVCC_FLAGS += -lineinfo -Xptxas -v
check: $(TARGET)

compatible: NVCC_FLAGS = -O3 -arch=sm_50 -gencode=arch=compute_50,code=sm_50 \
                         -gencode=arch=compute_60,code=sm_60 \
                         -gencode=arch=compute_70,code=sm_70 \
                         -gencode=arch=compute_75,code=sm_75 \
                         -Xcompiler -fopenmp -lpthread
compatible: $(TARGET)

clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f core*

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
	@for size in 256 512 1024 2048 4096 8192 16384; do \
		for alg in 1 2 3; do \
			if [ $$size -gt 4096 ] && [ $$alg -eq 1 ]; then continue; fi; \
			if [ $$alg -eq 1 ]; then threads=4; else threads=1; fi; \
			echo -n "$${size}x$${size} | "; \
			if [ $$alg -eq 1 ]; then echo -n "CPU      | "; \
			elif [ $$alg -eq 2 ]; then echo -n "GPU      | "; \
			else echo -n "GPUsm    | "; fi; \
			LC_NUMERIC=C ./$(TARGET) $$size $$threads $$alg 2>/dev/null | \
			grep "Tiempo de cómputo:" | \
			awk '{printf "%.9f | ", $$4}'; \
			LC_NUMERIC=C ./$(TARGET) $$size $$threads $$alg 2>/dev/null | \
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

.PHONY: all clean test benchmark profile help debug check compatible