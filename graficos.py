import matplotlib.pyplot as plt
import re
import os

if not os.path.exists('graficos'):
    os.makedirs('graficos')

logs = {"CPU": {"n": [], "t": []}, "GPU": {"n": [], "t": []}, "GPUsm": {"n": [], "t": []}}

with open("resultados.txt", "r") as f:
    for line in f:
        match = re.search(r"(\d+)x\d+\s*\|\s*(\w+)\s*\|\s*([\d,.]+)", line)
        if match:
            n, alg, t_str = match.groups()
            t_val = float(t_str.replace(',', '.'))
            if alg in logs:
                logs[alg]["n"].append(int(n))
                logs[alg]["t"].append(t_val)

#Grafico 1:Tiempos (Escala Logaritmica)
plt.figure(figsize=(10, 6))
plt.plot(logs["CPU"]["n"], logs["CPU"]["t"], 'o-', label='CPU')
plt.plot(logs["GPU"]["n"], logs["GPU"]["t"], 's-', label='GPU Basica')
plt.plot(logs["GPUsm"]["n"], logs["GPUsm"]["t"], '^-', label='GPU Shared')

plt.yscale('log')
plt.title('Tiempos de Ejecución')
plt.xlabel('Tamaño de matriz (N)')
plt.ylabel('Segundos')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('graficos/tiempo_vs_n.png')
plt.close()

#Grafico 2:Speedup
n_comun = sorted(list(set(logs["CPU"]["n"]) & set(logs["GPU"]["n"])))

speedup_gpu = []
speedup_gpu_sm = []

for n in n_comun:
    # Obtenemos el indice para este N en cada lista
    idx_cpu = logs["CPU"]["n"].index(n)
    idx_gpu = logs["GPU"]["n"].index(n)
    idx_sm = logs["GPUsm"]["n"].index(n)
    
    t_cpu = logs["CPU"]["t"][idx_cpu]
    speedup_gpu.append(t_cpu / logs["GPU"]["t"][idx_gpu])
    speedup_gpu_sm.append(t_cpu / logs["GPUsm"]["t"][idx_sm])

plt.figure(figsize=(10, 6))
plt.plot(n_comun, speedup_gpu, 's-', color='orange', label='Speedup GPU')
plt.plot(n_comun, speedup_gpu_sm, '^-', color='green', label='Speedup GPUsm')

plt.title('Speedup Relativo')
plt.xlabel('Tamaño de matriz (N)')
plt.ylabel('Factor de aceleración (x)')
plt.legend()
plt.grid(True)
plt.savefig('graficos/speedup_vs_n.png')
plt.close()

print("Gráficos generados correctamente.")