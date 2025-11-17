import numpy as np
import random
import pickle
import os

class AgenteQLearning:
    """
    Agente que aprende usando Q-Learning
    """

    def __init__(self, num_estados, num_acciones, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Inicializa el agente

        Args:
            num_estados: Número total de estados
            num_acciones: Número de acciones
            alpha: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon: Tasa de exploración
        """
        self.num_estados = num_estados
        self.num_acciones = num_acciones
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.tabla_q = np.zeros((num_estados, num_acciones))

    def elegir_accion(self, estado_indice, entrenar=True):
        """Elige una acción usando política ε-greedy"""
        if entrenar and random.random() < self.epsilon:
            return random.randint(0, self.num_acciones - 1)
        else:
            return np.argmax(self.tabla_q[estado_indice])

    def actualizar_q(self, estado, accion, recompensa, siguiente_estado, terminado):
        """Actualiza la tabla Q"""
        q_actual = self.tabla_q[estado, accion]

        if terminado:
            q_siguiente_max = 0
        else:
            q_siguiente_max = np.max(self.tabla_q[siguiente_estado])

        nuevo_q = q_actual + self.alpha * (recompensa + self.gamma * q_siguiente_max - q_actual)
        self.tabla_q[estado, accion] = nuevo_q

    def guardar_modelo(self, ruta='modelo_agente.pkl'):
        """
        Guarda la tabla Q y los parámetros del agente en un archivo .pkl
        El nombre del archivo es incremental para no sobrescribir modelos anteriores
        
        Args:
            ruta: Ruta base del archivo donde guardar el modelo
        """
        base_name, ext = os.path.splitext(ruta)
        contador = 1
        
        while os.path.exists(f"{base_name}_{contador}{ext}"):
            contador += 1
        
        ruta_final = f"{base_name}_{contador}{ext}"
        
        modelo = {
            'tabla_q': self.tabla_q,
            'num_estados': self.num_estados,
            'num_acciones': self.num_acciones,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
        
        with open(ruta_final, 'wb') as f:
            pickle.dump(modelo, f)
        
        print(f"Modelo guardado en: {ruta_final}")
        return ruta_final

    def cargar_modelo(self, ruta='modelo_agente.pkl'):
        """
        Carga la tabla Q y los parámetros del agente desde un archivo .pkl
        
        Args:
            ruta: Ruta del archivo desde donde cargar el modelo
        """
        if not os.path.exists(ruta):
            print(f"No se encontró el archivo: {ruta}")
            return False
        
        with open(ruta, 'rb') as f:
            modelo = pickle.load(f)
        
        self.tabla_q = modelo['tabla_q']
        self.num_estados = modelo['num_estados']
        self.num_acciones = modelo['num_acciones']
        self.alpha = modelo['alpha']
        self.gamma = modelo['gamma']
        self.epsilon = modelo['epsilon']
        
        print(f"Modelo cargado desde: {ruta}")
        return True
