# Librerias Necesarias
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image, ImageDraw, ImageOps
import queue
import math
import heapq
from collections import deque
import sys
import time
from scipy.spatial.distance import cdist

# ___________________________________________________________________________________
# Funciones
def leerImagen(nombre_imagen):
    imagen = cv2.imread(nombre_imagen)
    return imagen

def generarMatrizRutas (imagen):
  matriz_rutas = np.zeros((imagen.shape[0], imagen.shape[1]))
  for i in range(imagen.shape[0]):
    for j in range(imagen.shape[1]):
      if (imagen[i, j] < (240, 240, 240)).all():
        matriz_rutas[i, j] = 1
  return matriz_rutas

def leerPuntos(archivo):
    lista_puntos = []
    with open(archivo, 'r') as file:
        for linea in file:
            datos = linea.strip().split(', ')
            codigo_emergencia, x, y, tipo_vehiculo = datos
            lista_puntos.append((codigo_emergencia, int(x), int(y), tipo_vehiculo))

    return lista_puntos

def obtenerPuntoMasCercano(imagen, puntos):
    matriz = generarMatrizRutas(imagen)
    a, b = puntos

    punto_mas_cercano = None
    distancia_minima = float('inf')

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if matriz[i][j] == 1:
                distancia_actual = math.sqrt((j - b) ** 2 + (i - a) ** 2)
                if distancia_actual < distancia_minima:
                    distancia_minima = distancia_actual
                    punto_mas_cercano = (i, j)

    if punto_mas_cercano is not None:
        print(f"Encontrado punto más cercano en: {punto_mas_cercano}")
        return punto_mas_cercano
    else:
        print("No se encontró un punto navegable cercano.")
        return None

def encontrar_camino_cercano(matriz, x, y):
    if matriz[y][x] == 1:
        return x, y  # La posición original ya tiene camino

    visitados = set()
    cola = deque([(x, y)])

    while cola:
        x_actual, y_actual = cola.popleft()

        # Revisar puntos vecinos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x_vecino, y_vecino = x_actual + dx, y_actual + dy

            # Verificar si el vecino está dentro de los límites de la matriz y no ha sido visitado
            if 0 <= x_vecino < len(matriz[0]) and 0 <= y_vecino < len(matriz) and (x_vecino, y_vecino) not in visitados:
                if matriz[y_vecino][x_vecino] == 1:
                    return x_vecino, y_vecino  # Encontramos un punto con camino

                visitados.add((x_vecino, y_vecino))
                cola.append((x_vecino, y_vecino))

    return None  # No se encontró un punto con camino

def asignarVehiculos(despliegue, emergencia):
    asignaciones = {}
    tipos_emergencia_encontrados = set()

    for eme_line, eme_x, eme_y, eme_tipo in emergencia:
        if eme_tipo in tipos_emergencia_encontrados:
            continue

        for desp_tipo, desp_x, desp_y, desp_codigo in despliegue:
            if eme_tipo == desp_tipo:
                clave = (eme_line, eme_x, eme_y, eme_tipo)
                if clave not in asignaciones:
                    asignaciones[clave] = []
                asignaciones[clave].append([desp_x, desp_y, desp_codigo])

        # Agregar el tipo de emergencia a los encontrados
        tipos_emergencia_encontrados.add(eme_tipo)

    return asignaciones

def procesarResultado(resultado):
    coordenadas_emergencia = []
    coordenadas_carros = []

    for clave, lista_carros in resultado.items():
        eme_line, eme_x, eme_y, eme_tipo = clave
        coordenadas_emergencia.append((eme_x, eme_y))

        # Obtener las coordenadas de los carros asociados
        carros = [(x, y) for x, y, _ in lista_carros]
        coordenadas_carros.append(carros)

    return coordenadas_emergencia, coordenadas_carros

def bfs(matriz_rutas, inicio):
    cola = queue.Queue()
    visited = np.zeros_like(matriz_rutas)

    cola.put(inicio)
    visited[inicio] = 1

    while not cola.empty():
        x, y = cola.get()

        if matriz_rutas[x, y] == 1:
            return x, y  # Se encontró un píxel negro

        # Agregar vecinos no visitados
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < matriz_rutas.shape[0] and 0 <= ny < matriz_rutas.shape[1] and not visited[nx, ny]:
                    cola.put((nx, ny))
                    visited[nx, ny] = 1

    return None

def reubicarCoordenadas(lista_puntos, matriz_rutas):
    lista_puntosN = []
    for i in range(len(lista_puntos)):
        codigo_emergencia, x, y, tipo_vehiculo = lista_puntos[i]
        if matriz_rutas[x, y] == 1:
            # La posición original ya es válida, no es necesario reubicar
            lista_puntosN.append((codigo_emergencia, x, y, tipo_vehiculo))
        else:
            # Usar BFS para encontrar el píxel negro más cercano
            nuevo_x, nuevo_y = bfs(matriz_rutas, (x, y))
            # Actualizar las coordenadas
            lista_puntosN.append((codigo_emergencia, nuevo_x, nuevo_y, tipo_vehiculo))
    return lista_puntosN

    # (Tu código actual)

def dijkstra(grafo, inicio):
    # Distancias iniciales son infinitas
    distancias = {nodo: float('infinity') for nodo in grafo}
    # La distancia al nodo de inicio es 0
    distancias[inicio] = 0

    # La cola de prioridad de todos los nodos accesibles
    cola = [(0, inicio)]

    # Un set para rastrear todos los nodos visitados
    visitados = set()

    # Predecesores para rastrear el camino
    predecesores = {nodo: None for nodo in grafo}

    while cola:
        # Obtiene el nodo con la menor distancia
        distancia_actual, nodo_actual = heapq.heappop(cola)

        # Si el nodo ya ha sido visitado, lo ignoramos
        if nodo_actual in visitados:
            continue

        visitados.add(nodo_actual)

        # Consideramos los vecinos del nodo actual
        for vecino, peso in grafo[nodo_actual].items():
            distancia = distancia_actual + peso

            # Solo actualizamos la distancia del vecino si encontramos un camino más corto
            if distancia < distancias[vecino]:
                distancias[vecino] = distancia
                predecesores[vecino] = nodo_actual
                heapq.heappush(cola, (distancia, vecino))

    return distancias, predecesores 

def reconstruir_camino(predecesores, inicio, fin):
    camino = []
    nodo_actual = fin

    # Verificar si hay una ruta válida
    if predecesores[nodo_actual] is None:
        return None  # No hay ruta válida

    while nodo_actual != inicio:
        camino.append(nodo_actual)
        nodo_actual = predecesores[nodo_actual]

    camino.append(inicio)
    camino.reverse()
    return camino


def generarGrafo(matriz):
    pesos_ciudad = dict()
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if matriz[i, j] == 1:
                # Calculate the neighbors considering the matrix boundaries and connectivity (value is 1)
                vecinos = {(k, s): 1 if not (k == i or s == j) else 1.4 for k in range(i-1, i+2)
                           for s in range(j-1, j+2)
                           if 0 <= k < matriz.shape[0] and 0 <= s < matriz.shape[1] and matriz[k, s] == 1}
                # Remove self from neighbors
                del vecinos[(i, j)]
                pesos_ciudad[(i, j)] = vecinos
    return pesos_ciudad

def construir_matriz_distancia(matriz):
    altura, anchura = matriz.shape
    # Crear una matriz de coordenadas de píxeles
    x, y = np.meshgrid(range(anchura), range(altura))
    coordenadas = np.column_stack((x.ravel(), y.ravel()))

    # Calcular la matriz de distancia
    matriz_distancia = cdist(coordenadas, coordenadas, 'euclidean')
    
    return matriz_distancia
    
def asignar_codigos(despliegue, emergencia):
    asignaciones = {}

    for eme_line, eme_x, eme_y, eme_tipo in emergencia:
        for desp_tipo, desp_x, desp_y, desp_codigo in despliegue:
            if eme_tipo == desp_tipo:
                clave = (eme_line, eme_tipo)
                if clave not in asignaciones:
                    asignaciones[clave] = []
                asignaciones[clave].append(desp_codigo)

    return asignaciones

def marcar_puntos(imagen, puntos, color):
    for _, x, y, _ in puntos:
        cv2.circle(imagen, (y, x), 5, color, -1)

def generar_imagen(despliegue, emergencia, imagen):
    imagen_inicial = imagen.copy()

    # Marcar posiciones de camiones de bomberos (rojo)
    camiones = [punto for punto in despliegue if punto[0] == 'B']
    marcar_puntos(imagen_inicial, camiones, (0, 0, 255))

    # Marcar posiciones de patrullas de policía (verde)
    patrullas = [punto for punto in despliegue if punto[0] == 'P']
    marcar_puntos(imagen_inicial, patrullas, (0, 255, 0))

    # Marcar posiciones de ambulancias (azul)
    ambulancias = [punto for punto in despliegue if punto[0] == 'A']
    marcar_puntos(imagen_inicial, ambulancias, (255, 0, 0))

    # Marcar sitios de emergencias (fucsia)
    marcar_puntos(imagen_inicial, emergencia, (255, 0, 255))

    return imagen_inicial

def generar_imagen_atencion(despliegue, emergencia, coordenadas_carros, grafo, imagen):
    imagen_despues = generar_imagen(despliegue, emergencia, imagen.copy())

    for vehiculos in coordenadas_carros:
        for vehiculo in vehiculos:
            # Obtener la ruta desde el vehículo hasta la emergencia
            distancias, predecesores = dijkstra(grafo, vehiculo)
            camino = reconstruir_camino(predecesores, vehiculo, emergencia)

            # Marcar la ruta del vehículo con el color correspondiente
            color_ruta = obtener_color_vehiculo(vehiculo)
            marcar_ruta(imagen_despues, camino, color_ruta)

    return imagen_despues

def calcular_rutas_emergencia(coordenadas_emergencia, coordenadas_carros, grafo):
    rutas_emergencia = []
    for emergencia in coordenadas_emergencia:
        for vehiculos in coordenadas_carros:
            print(f"\nEmergencia: {emergencia}, Vehículos: {vehiculos}")
            for vehiculo in vehiculos:
                distancias, predecesores = dijkstra(grafo, vehiculo)
                camino = reconstruir_camino(predecesores, vehiculo, emergencia)
                print(f"Ruta más corta desde {vehiculo} hasta {emergencia}: {camino}") # CORREGIR
                rutas_emergencia.append(camino)
    return rutas_emergencia

def dibujar_ruta(imagen, ruta):
    for i in range(len(ruta) - 1):
        punto_inicial = ruta[i]
        punto_final = ruta[i + 1]
        cv2.line(imagen, punto_inicial, punto_final, (0, 255, 255), 2)  # Color amarillo, grosor 2

def crear_imagen_con_rutas(imagen_original,ruta,nombre):
    # Cargar la imagen original
    imagen = cv2.imread(imagen_original)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # Convertir a RGB para PIL
    imagen_pil = Image.fromarray(imagen)
    draw = ImageDraw.Draw(imagen_pil)

    for i in ruta:
        if i!=None:
            draw.line(i, fill=(255, 0, 0), width=-1)

    # Guardar o mostrar la imagen
    imagen_pil.save(nombre)

def dibujar_puntos(imagen, puntos, color):
    for tipo_vehiculo, y, x, _ in puntos:
        cv2.circle(imagen, (x, y), 5, color[tipo_vehiculo], -1)

def dibujar_emergencias(imagen, emergencias, color):
    for _, x, y, tipo_vehiculo in emergencias:
        cv2.circle(imagen, (x, y), 5, color, -1)

def crear_imagen_despliegue(despliegue, emergencia, imagen, nombre_imagen):
    imagen_despliegue = imagen.copy()

    # Definir colores para los vehículos y emergencias
    color_vehiculo = {'B': (0, 0, 255), 'P': (0, 255, 0), 'A': (255, 0, 0)}
    color_emergencia = (255, 0, 255)

    # Dibujar puntos de despliegue
    dibujar_puntos(imagen_despliegue, despliegue, color_vehiculo)

    # Dibujar puntos de emergencia
    dibujar_emergencias(imagen_despliegue, emergencia, color_emergencia)

    # Guardar la imagen
    cv2.imwrite(nombre_imagen, imagen_despliegue)


#___________________________________________________________________________________
# MAIN
def main():
    # Verifica que se hayan pasado el número correcto de argumentos
    if len(sys.argv) != 4:
        print("Uso: python SAIAE.py nombre_imagen.jpeg despliegue.txt emergencias.txt")
        sys.exit(1)

    # Guarda los argumentos en variables
    nombre_imagen = sys.argv[1]
    arg_despliegue = sys.argv[2]
    arg_emergencias = sys.argv[3]

    ti = time.time()
    # Lectura de archivos
    despliegue = leerPuntos(arg_despliegue)
    emergencias = leerPuntos(arg_emergencias)
    imagen = leerImagen(nombre_imagen)
    matriz=generarMatrizRutas(imagen)
    
    # Reubicacion de emergencias y vehiculos
    EmergenciaR = reubicarCoordenadas(emergencias, matriz)
    despliegueR = reubicarCoordenadas(despliegue, matriz) 

    # Primer Output
    imagen_2 = generar_imagen(despliegueR, EmergenciaR, imagen)
    if nombre_imagen[-5:]==".jpeg":
        name = nombre_imagen[:-5]
        cv2.imwrite(f'{nombre_imagen[:-5]}_despliegue.jpeg', imagen_2)
    else:
        name = nombre_imagen[:-4]
        cv2.imwrite(f'{nombre_imagen[:-4]}_despliegue.jpeg', imagen_2)
    

    # Segundo Output
    resultadoC = asignar_codigos(despliegueR, EmergenciaR)
    with open('atención.txt', 'w') as archivo:
        for (eme_line, eme_tipo), desp_codigos in resultadoC.items():
            archivo.write(f"{eme_line}, {eme_tipo}: {desp_codigos}\n")

    resultadoV = asignarVehiculos(despliegueR, EmergenciaR)
    with open('nodos.txt', 'w') as archivo:
        for (eme_line,  eme_x, eme_y,eme_tipo), asignacion in resultadoV.items():
            archivo.write(f"{eme_line}, {eme_x}, {eme_y}, {eme_tipo}, {asignacion}\n")
            
    
    grafo = generarGrafo(matriz)
    resultado = asignarVehiculos(despliegueR, EmergenciaR)

    coordenadas_emergencia, coordenadas_carros = procesarResultado(resultado)
    rutas = calcular_rutas_emergencia(coordenadas_emergencia, coordenadas_carros, grafo)

    # Tercer Output
    imagen_despliegue = name + '_despliegue.jpeg'
    nombre_img_atencion = name + '_atencion.jpeg'
    crear_imagen_con_rutas(imagen_despliegue, rutas, nombre_img_atencion)

    print(f"Tiempo de ejecución: {time.time() - ti} segundos")
    
if __name__ == "__main__":
    main()
