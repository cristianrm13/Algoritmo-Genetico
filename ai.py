#a1 c3 m1 p2

import math
import random
import os
from sympy import symbols, lambdify
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tabulate import tabulate
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ln[1+abs(x**2)]+2cosx
# sin(x)
formula = "log(1 + Abs(x**2)) + 2 * cos(x)"

def calcularFuncion(funcion, valorX):
    x = symbols('x')
    ex = lambdify(x, funcion, 'numpy')
    resultado = ex(valorX)
    return resultado

def calcularX(numGenerado, limInferior, delta):
    valorX = limInferior + numGenerado * delta
    return valorX

def generarPobInicial(pobInicial, numeroRango, numeroBits, limInferior, delta):
    pobGeneral = []
    for i in range(pobInicial):
        numGenerado = random.randint(1, numeroRango)
        numGeneradoBin = bin(numGenerado)[2:].zfill(numeroBits)
        valorX = calcularX(numGenerado, limInferior, delta)
        valorY = calcularFuncion(formula, valorX)
        individuo = {'id': i+1, 'binario': numGeneradoBin, 'i': numGenerado, 'x': valorX, 'y': valorY}
        pobGeneral.append(individuo)
    return pobGeneral

def calcularDatos(limInferior, limSuperior, resolucion):
    rango = limSuperior - limInferior
    puntos = math.ceil(rango / resolucion) + 1
    numeroBits = math.ceil(math.log2(puntos))
    delta = rango / ((2 ** numeroBits) - 1)
    numeroRango = 2 ** numeroBits
    return delta, numeroBits, numeroRango

def formarParejas(poblacion):
    parejas = []
    n = len(poblacion)
    for individuo in poblacion:
        m = random.randint(1, n)
        indices = random.sample(range(n), m)
        indices = [i for i in indices if i != individuo['id'] - 1]  # Omitir el mismo individuo
        for indice in indices:
            parejas.append((individuo, poblacion[indice]))
    return parejas

def cruzar(individuo1, individuo2, numeroBits):
    puntosDeCruza = [numeroBits // 3, 2 * numeroBits // 3]  # Múltiples puntos de cruza fijos
    bin1 = individuo1['binario']
    bin2 = individuo2['binario']
    nuevoBinario1 = bin1[:puntosDeCruza[0]] + bin2[puntosDeCruza[0]:puntosDeCruza[1]] + bin1[puntosDeCruza[1]:]
    nuevoBinario2 = bin2[:puntosDeCruza[0]] + bin1[puntosDeCruza[0]:puntosDeCruza[1]] + bin2[puntosDeCruza[1]:]
    return nuevoBinario1, nuevoBinario2

""" def mutar(individuo, probMutacionGen):
    binarioSeparado = list(individuo)
    # Iterar sobre cada gen (bit) del individuo
    for i in range(len(binarioSeparado)):
        # Evaluar si el gen debe mutar basado en la probabilidad probMutacionGen
        if random.random() < probMutacionGen: #es la probabilidad de que cada gen del individuo muta.
            # Si el gen debe mutar, seleccionar un índice aleatorio en la cadena entre 0 y 1
            j = random.randint(0, len(binarioSeparado) - 1)
            # Intercambiar el gen actual con el gen en el índice seleccionado
            binarioSeparado[i], binarioSeparado[j] = binarioSeparado[j], binarioSeparado[i]
            # Convertir la lista de vuelta a una cadena de bits y devolver el individuo mutado
    return ''.join(binarioSeparado) """

""" def mutar(individuo, probMutacionGen):
    binarioSeparado = list(individuo)
    for i in range(len(binarioSeparado)):
        if random.random() < probMutacionGen:
            # Negar el bit
            binarioSeparado[i] = '1' if binarioSeparado[i] == '0' else '0'
    return ''.join(binarioSeparado) """
    
def mutar(individuo, probMutacionGen):
    # Convertir el individuo (cadena de bits) en una lista de caracteres para facilitar la manipulación
    binarioSeparado = list(individuo)
    
    # Iterar sobre cada gen (bit) del individuo
    for i in range(len(binarioSeparado)):
        # Evaluar si el gen debe mutar basado en la probabilidad probMutacionGen
        # Generar un número aleatorio entre 0 y 1 y comparar con probMutacionGen
        if random.random() < probMutacionGen:
            # Si el número aleatorio es menor que probMutacionGen, el gen debe mutar
            # Intercambiar el valor del bit actual: si es '0' cambiarlo a '1' y viceversa
            binarioSeparado[i] = '1' if binarioSeparado[i] == '0' else '0'
    
    # Convertir la lista de vuelta a una cadena de bits y devolver el individuo mutado
    return ''.join(binarioSeparado)

def crearNuevosIndividuos(individuo1, individuo2, limInferior, delta):
    numeroDec1 = int(individuo1, 2)
    numeroDec2 = int(individuo2, 2)
    x1 = calcularX(numeroDec1, limInferior, delta)
    x2 = calcularX(numeroDec2, limInferior, delta)
    y1 = calcularFuncion(formula, x1)
    y2 = calcularFuncion(formula, x2)

    nuevoIndividuo1 = {'id': None, 'binario': individuo1, 'i': numeroDec1, 'x': x1, 'y': y1}
    nuevoIndividuo2 = {'id': None, 'binario': individuo2, 'i': numeroDec2, 'x': x2, 'y': y2}

    return nuevoIndividuo1, nuevoIndividuo2

def podarPoblacion(poblacion, pobMaxima, tipoPm):
    pobUnica = {individuo['i']: individuo for individuo in poblacion}.values()  # Eliminar duplicados basados en el índice 'i'
    pobOrdenada = sorted(pobUnica, key=lambda x: x['y'], reverse=(tipoPm == "Maximizacion"))
    mejorIndividuo = pobOrdenada[0]
    if len(pobOrdenada) > 1:
        restoPoblacion = random.sample(pobOrdenada[1:], min(pobMaxima - 1, len(pobOrdenada) - 1))  # Mantener los mejores individuos
    else:
        restoPoblacion = []
    
    nvPob = [mejorIndividuo] + restoPoblacion
    
    return nvPob

def graficarGeneracion(generacion, poblacion, limInferior, limSuperior, tipoPm):
    plt.clf()
    plt.xlim(limInferior, limSuperior)
    plt.title(f'Generacion {generacion}, f(x)={formula}')
    plt.xlabel('X')
    plt.ylabel('f(X)')

    xValues = [individuo['x'] for individuo in poblacion]
    yValues = [individuo['y'] for individuo in poblacion]
    plt.scatter(xValues, yValues, label="individuos", s=90, c="coral", alpha=0.4)

    if tipoPm == "Maximizacion":
        mejorIndividuo = max(poblacion, key=lambda individuo: individuo['y'])
        peorIndividuo = min(poblacion, key=lambda individuo: individuo['y'])
    else:
        mejorIndividuo = min(poblacion, key=lambda individuo: individuo['y'])
        peorIndividuo = max(poblacion, key=lambda individuo: individuo['y'])

    xFunc = np.linspace(limInferior, limSuperior, 200)
    x = symbols('x')
    ex = lambdify(x, formula, 'numpy')
    yFunc = ex(xFunc)
    plt.plot(xFunc, yFunc, color='black')
    plt.scatter(mejorIndividuo['x'], mejorIndividuo['y'], c='tab:green', label='Mejor Individuo', s=90)
    plt.scatter(peorIndividuo['x'], peorIndividuo['y'], c='tab:cyan', label='Peor Individuo', s=90)
    plt.legend()

    folderPath = 'Historial'
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    plt.savefig(os.path.join(folderPath, f'Generacion_{generacion}.png'))
    plt.close()

def graficarEstadisticas(generaciones, mejoresY, peoresY, promediosY):
    plt.figure()
    plt.plot(generaciones, mejoresY, label='Mejor Individuo')
    plt.plot(generaciones, peoresY, color='crimson', label='Peor Individuo')
    plt.plot(generaciones, promediosY, label='Media de individuo')

    plt.title('Evolución del fitness')
    plt.xlabel('Generación')
    plt.ylabel('Valor de la Función Objetivo')
    plt.legend()

    folderPath = 'MedianaAptitud'
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    plt.savefig(os.path.join(folderPath, 'MdGeneraciones.png'))
    plt.close()

def crearVideo():
    folderPath = 'Historial'
    videoPath = 'AGr_poblacion.mp4'
    images = [img for img in os.listdir(folderPath) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    frame = cv2.imread(os.path.join(folderPath, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(videoPath, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(folderPath, image)))
    video.release()

def validar_entradas():
    try:
        limInferior = float(entry_limInferior.get())
        limSuperior = float(entry_limSuperior.get())
        resolucion = float(entry_resolucion.get())
        pobInicial = int(entry_pobInicial.get())
        pobMaxima = int(entry_pobMaxima.get())
        probMutacionInd = float(entry_probMutacionInd.get())
        probMutacionGen = float(entry_probMutacionGen.get())
        numGeneraciones = int(entry_numGeneraciones.get())

        if limInferior >= limSuperior:
            raise ValueError("ej: LI: 10, LS: 50.")
        if not (0 < resolucion < 1):
            raise ValueError("Mayor que 0 y menor que 1.")
        if pobInicial <= 0:
            raise ValueError("Mayor que 0.")
        if pobMaxima <= 0:
            raise ValueError("Mayor que 0.")
        if not (0 <= probMutacionInd <= 1):
            raise ValueError("Rango entre 0 a 1.")
        if not (0 <= probMutacionGen <= 1):
            raise ValueError("Rango entre 0 a 1.")
        if numGeneraciones <= 0:
            raise ValueError("ej: 100")
        
        return True
    except ValueError as e:
        messagebox.showerror("Error de validación", str(e))
        return False

def algoritmoGenetico(limInferior, limSuperior, resolucion, pobInicial, pobMaxima, probMutacionInd, probMutacionGen, numGeneraciones, tipoPm, output_text, tree):
    delta, numeroBits, numeroRango = calcularDatos(limInferior, limSuperior, resolucion)
    pobGeneral = generarPobInicial(pobInicial, numeroRango, numeroBits, limInferior, delta)

    generaciones = []
    mejoresY = []
    peoresY = []
    promediosY = []
    estadisticasGenerales = []

    output_text.insert(tk.END, "\nPoblación 0:\n")
    for individuo in pobGeneral:
        output_text.insert(tk.END, f"id: {individuo['id']}, i: {individuo['i']}, binario: {individuo['binario']}, x: {round(individuo['x'], 3)}, f(x): {round(individuo['y'], 3)}\n")
    graficarGeneracion(0, pobGeneral, limInferior, limSuperior, tipoPm)

    for generacion in range(1, numGeneraciones + 1):
        output_text.insert(tk.END, f"\nGeneracion {generacion}:\n")
        parejas = formarParejas(pobGeneral)
        nvPob = []

        for pareja in parejas: # Cruzar dos individuos para generar dos nuevos binarios
            nuevoBinario1, nuevoBinario2 = cruzar(pareja[0], pareja[1], numeroBits) #son los nuevos individuos creados después del cruce.
                # Evaluar si el primer nuevo binario debe mutar basado en probMutacionInd
            if random.random() < probMutacionInd: # decide si el individuo muta
                # Si debe mutar, aplicar la función mutar con probMutacionGen
                nuevoBinario1 = mutar(nuevoBinario1, probMutacionGen)
                # Evaluar si el segundo nuevo binario debe mutar basado en probMutacionInd
            if random.random() < probMutacionInd: 
                # Si debe mutar, aplicar la función mutar con probMutacionGen
                nuevoBinario2 = mutar(nuevoBinario2, probMutacionGen)
                 # Crear nuevos individuos a partir de los binarios posiblemente mutados
            nuevoIndividuo1, nuevoIndividuo2 = crearNuevosIndividuos(nuevoBinario1, nuevoBinario2, limInferior, delta)
            # Agregar los nuevos individuos a la nueva población
            nvPob.append(nuevoIndividuo1)
            nvPob.append(nuevoIndividuo2)

        for i, individuo in enumerate(nvPob): # Asignar un ID único a cada nuevo individuo
            individuo['id'] = len(pobGeneral) + i + 1

        pobGeneral.extend(nvPob)
        
        # Evaluar el mejor y peor individuo de la población actual (antes de la poda)
        if tipoPm == "Maximizacion":
            mejorIndividuo = max(pobGeneral, key=lambda individuo: individuo['y'])
            peorIndividuo = min(pobGeneral, key=lambda individuo: individuo['y'])
        else:
            mejorIndividuo = min(pobGeneral, key=lambda individuo: individuo['y'])
            peorIndividuo = max(pobGeneral, key=lambda individuo: individuo['y'])

        promedioY = sum(individuo['y'] for individuo in pobGeneral) / len(pobGeneral)

        generaciones.append(generacion)
        mejoresY.append(mejorIndividuo['y'])
        peoresY.append(peorIndividuo['y'])
        promediosY.append(promedioY)

        estadisticasGenerales.append([generacion, mejorIndividuo['id'], round(mejorIndividuo['x'], 3), round(mejorIndividuo['y'], 3), mejorIndividuo['binario'],
                                      peorIndividuo['id'], round(peorIndividuo['x'], 3), round(peorIndividuo['y'], 3), peorIndividuo['binario'], round(promedioY, 3)])

        output_text.insert(tk.END, f"\nEstadísticas de la Generación {generacion}:\n")
        for individuo in pobGeneral:
            output_text.insert(tk.END, f"id: {individuo['id']}, i: {individuo['i']}, binario: {individuo['binario']}, x: {round(individuo['x'], 3)}, f(x): {round(individuo['y'], 3)}\n")
        output_text.insert(tk.END, f"Mejor individuo: id: {mejorIndividuo['id']}, x: {round(mejorIndividuo['x'], 3)}, f(x): {round(mejorIndividuo['y'], 3)}, binario: {mejorIndividuo['binario']}\n")
        output_text.insert(tk.END, f"Peor individuo: id: {peorIndividuo['id']}, x: {round(peorIndividuo['x'], 3)}, f(x): {round(peorIndividuo['y'], 3)}, binario: {peorIndividuo['binario']}\n")
        output_text.insert(tk.END, f"Promedio de aptitud: {round(promedioY, 3)}\n")

        # Actualizar la tabla con el mejor individuo de la generación actual
        tree.insert('', tk.END, values=(generacion, mejorIndividuo['id'], round(mejorIndividuo['x'], 3), round(mejorIndividuo['y'], 3), mejorIndividuo['binario']))

        # Podar la población
        pobGeneral = podarPoblacion(pobGeneral, pobMaxima, tipoPm)
        graficarGeneracion(generacion, pobGeneral, limInferior, limSuperior, tipoPm)

    graficarEstadisticas(generaciones, mejoresY, peoresY, promediosY)
    crearVideo()
    output_text.insert(tk.END, "\nEstadísticas Generales:\n")
    headers = ["Generación", "Mejor ID", "Mejor X", "Mejor f(X)", "Mejor Binario",
               "Peor ID", "Peor X", "Peor f(X)", "Peor Binario", "Promedio Generación"]
    table = tabulate(estadisticasGenerales, headers=headers, tablefmt="grid")
    output_text.insert(tk.END, f"{table}\n")

def ejecutar_algoritmo():
    if validar_entradas():
        limInferior = float(entry_limInferior.get())
        limSuperior = float(entry_limSuperior.get())
        resolucion = float(entry_resolucion.get())
        pobInicial = int(entry_pobInicial.get())
        pobMaxima = int(entry_pobMaxima.get())
        probMutacionInd = float(entry_probMutacionInd.get())
        probMutacionGen = float(entry_probMutacionGen.get())
        numGeneraciones = int(entry_numGeneraciones.get())
        tipoPm = combobox_tipoPm.get()

        output_text.delete(1.0, tk.END)
        tree.delete(*tree.get_children())
        algoritmoGenetico(limInferior, limSuperior, resolucion, pobInicial, pobMaxima, probMutacionInd, probMutacionGen, numGeneraciones, tipoPm, output_text, tree)

root = tk.Tk()
root.title("AG")
mainframe = ttk.Frame(root, padding="20 20 20 20")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(mainframe, text="Límite Inferior:      (ej: LI: 10)").grid(column=1, row=1, sticky=tk.W)
entry_limInferior = ttk.Entry(mainframe)
entry_limInferior.grid(column=2, row=1, sticky=(tk.W, tk.E))

ttk.Label(mainframe, text="Límite Superior:      (ej: LS: 50.)").grid(column=1, row=2, sticky=tk.W)
entry_limSuperior = ttk.Entry(mainframe)
entry_limSuperior.grid(column=2, row=2, sticky=(tk.W, tk.E))

ttk.Label(mainframe, text="Delta X:      (Rango entre 0 a 1)").grid(column=1, row=3, sticky=tk.W)
entry_resolucion = ttk.Entry(mainframe)
entry_resolucion.grid(column=2, row=3, sticky=(tk.W, tk.E))

ttk.Label(mainframe, text="Población Inicial:      (Mayor a 0)").grid(column=1, row=4, sticky=tk.W)
entry_pobInicial = ttk.Entry(mainframe)
entry_pobInicial.grid(column=2, row=4, sticky=(tk.W, tk.E))

ttk.Label(mainframe, text="Población Máxima:      (Mayor a Inicial)").grid(column=1, row=5, sticky=tk.W)
entry_pobMaxima = ttk.Entry(mainframe)
entry_pobMaxima.grid(column=2, row=5, sticky=(tk.W, tk.E))

ttk.Label(mainframe, text="Probabilidad de la Mutación del Individuo:      (Rango entre 0 a 1)").grid(column=1, row=6, sticky=tk.W)
entry_probMutacionInd = ttk.Entry(mainframe)
entry_probMutacionInd.grid(column=2, row=6, sticky=(tk.W, tk.E))

ttk.Label(mainframe, text="Probabilidad de la Mutación del Gen:      (Rango entre 0 a 1)").grid(column=1, row=7, sticky=tk.W)
entry_probMutacionGen = ttk.Entry(mainframe)
entry_probMutacionGen.grid(column=2, row=7, sticky=(tk.W, tk.E))

ttk.Label(mainframe, text="Número de Generaciones: ").grid(column=1, row=8, sticky=tk.W)
entry_numGeneraciones = ttk.Entry(mainframe)
entry_numGeneraciones.grid(column=2, row=8, sticky=(tk.W, tk.E))

ttk.Label(mainframe, text="Tipo de Problema:      (Maximizacion ó Minimizacion)").grid(column=1, row=9, sticky=tk.W)
combobox_tipoPm = ttk.Combobox(mainframe, values=["Maximizacion", "Minimizacion"])
combobox_tipoPm.grid(column=2, row=9, sticky=(tk.W, tk.E))
combobox_tipoPm.current(0)

ttk.Button(mainframe, text="Compilar", command=ejecutar_algoritmo).grid(column=2, row=10, sticky=tk.E)

output_text = scrolledtext.ScrolledText(mainframe, width=86, height=10, wrap=tk.WORD)
output_text.grid(column=1, row=11, columnspan=2, sticky=(tk.W, tk.E))

# Agregar el Treeview para mostrar el mejor individuo de cada generación
columns = ("Generación", "ID", "X", "f(X)", "BIN")
tree = ttk.Treeview(mainframe, columns=columns, show='headings')
for col in columns:
    tree.heading(col, text=col)
tree.grid(column=1, row=12, columnspan=2, sticky=(tk.W, tk.E))

for child in mainframe.winfo_children():
    child.grid_configure(padx=10, pady=5)

root.mainloop()
