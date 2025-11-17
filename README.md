# Aprendizaje por Refuerzo — Proyecto Q-Learning

## Descripción del entorno

El entorno implementado es un **GridWorld**: una cuadrícula bidimensional donde un agente debe encontrar un tesoro evitando trampas. El entorno genera posiciones aleatorias para trampas, tesoro e inicio en cada configuración. El agente observa el estado actual (posición en el grid) y puede moverse en las cuatro direcciones cardinales. Cada acción puede llevarlo a una celda vacía, una trampa (penalización), el tesoro (recompensa máxima) o fuera de los límites (penalización).

- **Estados:** Posiciones del agente en la cuadrícula.
- **Acciones:** Arriba, abajo, izquierda, derecha.
- **Recompensas:**
  - Tesoro: +100
  - Trampa: -100
  - Movimiento normal: -1
  - Salida de límites: -10

## Algoritmo usado

El agente utiliza **Q-Learning**, un algoritmo de aprendizaje por refuerzo off-policy. Q-Learning actualiza una tabla Q(s, a) que estima el valor esperado de tomar una acción a en un estado s y seguir la mejor política posible a partir de ahí. El agente explora el entorno usando una política ε-greedy (explora con probabilidad ε, explota con 1-ε), y actualiza sus valores Q tras cada transición usando la ecuación:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
$$

- **α (alpha):** Tasa de aprendizaje
- **γ (gamma):** Factor de descuento
- **ε (epsilon):** Tasa de exploración

## Comportamiento obtenido

Tras el entrenamiento, el agente aprende a navegar el grid evitando trampas y buscando el tesoro de forma eficiente. El comportamiento observado incluye:

- **Mejora progresiva:** El número de pasos por episodio disminuye y la tasa de éxito aumenta a medida que el agente aprende.
- **Exploración inicial:** El agente explora ampliamente al inicio, cometiendo errores y cayendo en trampas.
- **Explotación final:** Con el tiempo, el agente explota el conocimiento adquirido, encontrando el tesoro en menos pasos y evitando trampas.
- **Visualización:** El sistema incluye gráficas de recompensas, pasos y tasa de éxito, así como la trayectoria óptima encontrada por el agente.

Este proyecto demuestra cómo el aprendizaje por refuerzo permite a un agente aprender comportamientos óptimos en entornos desconocidos mediante prueba y error, usando únicamente señales de recompensa.
