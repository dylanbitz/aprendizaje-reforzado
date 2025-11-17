import numpy as np
import random

class GridWorld:
    """
    Entorno GridWorld donde un agente busca un tesoro evitando trampas
    """

    def __init__(self, size=10, num_trampas=10, max_pasos=200):
        """
        Inicializa el entorno

        Args:
            size: Tamaño del grid (size x size)
            num_trampas: Número de trampas
            max_pasos: Máximo número de pasos por episodio
        """
        self.size = size
        self.num_trampas = num_trampas
        self.max_pasos = max_pasos

        self.inicio = (0, 0)
        self.tesoro = (size-1, size-1)

        self.trampas = self._generar_trampas()

        self.posicion_agente = self.inicio
        self.pasos_actuales = 0

        # Acciones: 0=Arriba, 1=Abajo, 2=Izquierda, 3=Derecha
        self.acciones = [0, 1, 2, 3]

    def _generar_trampas(self):
        """Genera posiciones aleatorias para las trampas"""
        trampas = set()
        while len(trampas) < self.num_trampas:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) != self.inicio and (x, y) != self.tesoro:
                trampas.add((x, y))
        return trampas

    def reset(self):
        """Reinicia el entorno"""
        self.posicion_agente = self.inicio
        self.pasos_actuales = 0
        return self.posicion_agente

    def step(self, accion):
        """
        Ejecuta una acción

        Returns:
            nuevo_estado, recompensa, terminado, info
        """
        x, y = self.posicion_agente
        self.pasos_actuales += 1

        #posiciones
        if accion == 0:  # Arriba
            nueva_x, nueva_y = x - 1, y
        elif accion == 1:  # Abajo
            nueva_x, nueva_y = x + 1, y
        elif accion == 2:  # Izquierda
            nueva_x, nueva_y = x, y - 1
        else:  # Derecha
            nueva_x, nueva_y = x, y + 1

        if nueva_x < 0 or nueva_x >= self.size or nueva_y < 0 or nueva_y >= self.size:
            recompensa = -10
            terminado = False
            info = {"evento": "pared", "pasos": self.pasos_actuales}
            return self.posicion_agente, recompensa, terminado, info

        self.posicion_agente = (nueva_x, nueva_y)

        if self.posicion_agente == self.tesoro:
            recompensa = 100
            terminado = True
            info = {"evento": "tesoro", "pasos": self.pasos_actuales, "exito": True}
            return self.posicion_agente, recompensa, terminado, info

        if self.posicion_agente in self.trampas:
            recompensa = -100
            terminado = False
            info = {"evento": "trampa", "pasos": self.pasos_actuales}
            return self.posicion_agente, recompensa, terminado, info

        if self.pasos_actuales >= self.max_pasos:
            recompensa = -1
            terminado = True
            info = {"evento": "max_pasos", "pasos": self.pasos_actuales, "exito": False}
            return self.posicion_agente, recompensa, terminado, info

        recompensa = -1
        terminado = False
        info = {"evento": "paso", "pasos": self.pasos_actuales}
        return self.posicion_agente, recompensa, terminado, info

    def get_estado_grid(self):
        """Retorna el estado completo del grid para visualización"""
        return {
            'size': self.size,
            'agente': list(self.posicion_agente),
            'tesoro': list(self.tesoro),
            'trampas': [list(t) for t in self.trampas],
            'inicio': list(self.inicio),
            'pasos': self.pasos_actuales
        }

    def estado_a_indice(self, estado):
        """Convierte un estado (x, y) a un índice único"""
        x, y = estado
        return x * self.size + y

    def get_num_estados(self):
        """Retorna el número total de estados"""
        return self.size * self.size

    def get_num_acciones(self):
        """Retorna el número de acciones"""
        return len(self.acciones)
