import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def cargar_datos(archivo):
    datos = np.genfromtxt(archivo, delimiter=',')
    entradas = datos[:, :-1]
    salidas = datos[:, -1]
    return entradas, salidas


def evaluar_entrenar_perceptron(particion, tasa_aprendizaje, max_epocas, convergencia_criterio):
    entrenamiento_entradas = particion['entrenamiento_entradas']
    entrenamiento_salidas = particion['entrenamiento_salidas']
    entradas_prueba = particion['entradas_prueba']
    prueba_salidas = particion['prueba_salidas']   
    entradas_num = entrenamiento_entradas.shape[1]
    num_patrones = entrenamiento_entradas.shape[0]
    pesos = np.random.rand(entradas_num)
    bias = np.random.rand()
    
    epoca = 0
    convergencia = False
    
    while epoca < max_epocas and not convergencia:
        convergencia = True
        for i in range(num_patrones):
            entrada = entrenamiento_entradas[i]
            salida_deseada = entrenamiento_salidas[i]
            salida_obtenida = np.dot(pesos, entrada) + bias
            error = salida_deseada - salida_obtenida
            
            if abs(error) > convergencia_criterio:
                convergencia = False
                pesos += tasa_aprendizaje * error * entrada
                bias += tasa_aprendizaje * error
        
        epoca += 1
    salidas_predichas = np.sign(np.dot(entradas_prueba, pesos) + bias)
    accuracy = accuracy_score(prueba_salidas, salidas_predichas)
    
    return accuracy

def generar_particiones(entradas, salidas, num_particiones, porcentaje_entrenamiento):
    particiones = []
    for _ in range(num_particiones):
        entrenamiento_entradas, entradas_prueba, entrenamiento_salidas, prueba_salidas = train_test_split(
            entradas, salidas, test_size=(1 - porcentaje_entrenamiento), random_state=None)
        
        particion = {
            "entrenamiento_entradas": entrenamiento_entradas,
            "entradas_prueba": entradas_prueba,
            "entrenamiento_salidas": entrenamiento_salidas,
            "prueba_salidas": prueba_salidas
        }
        
        particiones.append(particion)
    
    return particiones


if __name__ == "__main__":
    entradas, salidas = cargar_datos("Practica 1 - Ejercicio 2/spheres2d70.csv")
    tasa_aprendizaje = 0.1
    max_epocas = 1000
    convergencia_criterio = 0.01
    num_particiones = 10
    porcentaje_entrenamiento = 0.8
    particiones = generar_particiones(entradas, salidas, num_particiones, porcentaje_entrenamiento)
    accuracies = []
    for i, particion in enumerate(particiones):
        accuracy = evaluar_entrenar_perceptron(particion, tasa_aprendizaje, max_epocas, convergencia_criterio)
        accuracies.append(accuracy)
        print(f"Partición {i + 1} - Accuracy: {accuracy}")
    promedio_accuracy = np.mean(accuracies)
    print(f"Promedio de accuracies: {promedio_accuracy}")
