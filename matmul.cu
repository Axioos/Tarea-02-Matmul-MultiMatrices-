#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <cuda_runtime.h>

// Estructura para pasar argumentos a los hilos de CPU
typedef struct {
    float *A;
    float *B;
    float *C;
    int n;
    int start_row;
    int end_row;
} ThreadData;

// Función para los hilos de CPU
void* matrix_multiply_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < data->n; k++) {
                sum += data->A[i * data->n + k] * data->B[k * data->n + j];
            }
            data->C[i * data->n + j] = sum;
        }
    }
    
    return NULL;
}

// Versión CPU multicore
float matrix_multiply_cpu(float *A, float *B, float *C, int n, int num_threads) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadData* thread_data = (ThreadData*)malloc(num_threads * sizeof(ThreadData));
    
    int rows_per_thread = n / num_threads;
    
    // Crear hilos
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].A = A;
        thread_data[t].B = B;
        thread_data[t].C = C;
        thread_data[t].n = n;
        thread_data[t].start_row = t * rows_per_thread;
        thread_data[t].end_row = (t == num_threads - 1) ? n : (t + 1) * rows_per_thread;
        
        pthread_create(&threads[t], NULL, matrix_multiply_thread, &thread_data[t]);
    }
    
    // Esperar a que terminen los hilos
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    free(threads);
    free(thread_data);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                    (end.tv_nsec - start.tv_nsec) / 1e9;
    return (float)elapsed;
}

// Kernel GPU básico
__global__ void matrix_multiply_gpu(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Kernel GPU con memoria compartida
__global__ void matrix_multiply_gpu_sm(float *A, float *B, float *C, int n) {
    // Tamaño del bloque (asumimos bloque cuadrado)
    const int BLOCK_SIZE = 16;
    
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Procesamiento por tiles
    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // Cargar datos a memoria compartida
        int load_row = row;
        int load_col = tile * BLOCK_SIZE + threadIdx.x;
        
        if (load_row < n && load_col < n) {
            sA[threadIdx.y][threadIdx.x] = A[load_row * n + load_col];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        load_row = tile * BLOCK_SIZE + threadIdx.y;
        load_col = col;
        
        if (load_row < n && load_col < n) {
            sB[threadIdx.y][threadIdx.x] = B[load_row * n + load_col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Calcular producto parcial
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Escribir resultado si estamos dentro de los límites
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Versión GPU básica
float matrix_multiply_gpu_wrapper(float *h_A, float *h_B, float *h_C, int n) {
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 
                  (n + blockSize.y - 1) / blockSize.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Launch kernel
    matrix_multiply_gpu<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    
    // Record stop event and wait
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / 1000.0f; // Convertir a segundos
}

// Versión GPU con memoria compartida
float matrix_multiply_gpu_sm_wrapper(float *h_A, float *h_B, float *h_C, int n) {
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch (bloques de 16x16 para memoria compartida)
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 
                  (n + blockSize.y - 1) / blockSize.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Launch kernel con memoria compartida
    matrix_multiply_gpu_sm<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    
    // Record stop event and wait
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / 1000.0f; // Convertir a segundos
}

// Función para inicializar matrices
void initialize_matrices(float *A, float *B, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }
}

// Función para verificar resultados (comparar CPU vs GPU)
int verify_results(float *C1, float *C2, int n, float epsilon) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(C1[i] - C2[i]) > epsilon) {
            printf("Diferencia en posición %d: CPU=%f, GPU=%f\n", i, C1[i], C2[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s <n> <nt> <ALG>\n", argv[0]);
        printf("  n:   tamaño de la matriz (n x n)\n");
        printf("  nt:  número de hilos CPU (solo para ALG=1)\n");
        printf("  ALG: algoritmo (1=CPU, 2=GPU, 3=GPUsm)\n");
        return 1;
    }
    
    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    int algorithm = atoi(argv[3]);
    
    if (n <= 0) {
        printf("Error: n debe ser positivo\n");
        return 1;
    }
    
    if (algorithm == 1 && num_threads <= 0) {
        printf("Error: nt debe ser positivo para CPU\n");
        return 1;
    }
    
    if (algorithm < 1 || algorithm > 3) {
        printf("Error: ALG debe ser 1, 2 o 3\n");
        return 1;
    }
    
    printf("========================================\n");
    printf("Multiplicación de matrices %dx%d\n", n, n);
    printf("Algoritmo: ");
    switch(algorithm) {
        case 1: printf("CPU (%d hilos)\n", num_threads); break;
        case 2: printf("GPU básica\n"); break;
        case 3: printf("GPU con memoria compartida\n"); break;
    }
    printf("========================================\n");
    
    // Asignar memoria
    size_t size = n * n * sizeof(float);
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);
    float *C_ref = NULL;
    
    // Inicializar matrices con valores aleatorios
    srand(time(NULL));
    initialize_matrices(A, B, n);
    
    float elapsed_time = 0.0f;
    
    // Ejecutar algoritmo seleccionado
    switch(algorithm) {
        case 1: // CPU
            elapsed_time = matrix_multiply_cpu(A, B, C, n, num_threads);
            break;
            
        case 2: // GPU básica
            elapsed_time = matrix_multiply_gpu_wrapper(A, B, C, n);
            break;
            
        case 3: // GPU con memoria compartida
            elapsed_time = matrix_multiply_gpu_sm_wrapper(A, B, C, n);
            break;
    }
    
    // Resultados
    printf("\nRESULTADOS:\n");
    printf("Tiempo de cómputo: %.6f segundos\n", elapsed_time);
    printf("Tasa de cómputo: %.2f GFLOPs\n", 
           (2.0 * n * n * n) / (elapsed_time * 1e9));
    printf("========================================\n");
    
    // Verificación opcional (comparar con CPU si no es CPU)
    /*
    if (algorithm != 1) {
        C_ref = (float*)malloc(size);
        float cpu_time = matrix_multiply_cpu(A, B, C_ref, n, 1); // Usar 1 hilo para verificación
        if (verify_results(C, C_ref, n, 1e-4)) {
            printf("✓ Resultados verificados correctamente\n");
        } else {
            printf("✗ Error en la verificación de resultados\n");
        }
        free(C_ref);
    }
    */
    
    // Liberar memoria
    free(A);
    free(B);
    free(C);
    
    return 0;
}