# -*- coding: utf-8 -*-
import numpy as np
import random

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

        # Tabla Q
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
