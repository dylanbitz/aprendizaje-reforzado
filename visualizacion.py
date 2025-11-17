import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os

class Visualizador:
    """
    Clase para crear visualizaciones del entrenamiento y desempeño del agente
    """
    
    def __init__(self, carpeta_salida='resultados/graficas'):
        """
        Inicializa el visualizador
        
        Args:
            carpeta_salida: Carpeta donde guardar las gráficas
        """
        self.carpeta_salida = carpeta_salida
        
        # Crear carpeta si no existe
        os.makedirs(carpeta_salida, exist_ok=True)
    
    def graficar_recompensas(self, recompensas, ventana=50, guardar=True, mostrar=False):
        """
        Grafica la evolución de las recompensas durante el entrenamiento
        
        Args:
            recompensas: Lista de recompensas por episodio
            ventana: Tamaño de ventana para promedio móvil
            guardar: Si guardar la gráfica
            mostrar: Si mostrar la gráfica
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodios = range(1, len(recompensas) + 1)
        
        # Gráfica de recompensas individuales (más transparente)
        ax.plot(episodios, recompensas, alpha=0.3, color='blue', 
                linewidth=0.5, label='Recompensa por episodio')
        
        # Promedio móvil
        if len(recompensas) >= ventana:
            promedio_movil = np.convolve(recompensas, 
                                        np.ones(ventana)/ventana, 
                                        mode='valid')
            episodios_promedio = range(ventana, len(recompensas) + 1)
            ax.plot(episodios_promedio, promedio_movil, 
                   color='red', linewidth=2, 
                   label=f'Promedio móvil (ventana={ventana})')
        
        # Línea de referencia en 0
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Configuración
        ax.set_xlabel('Episodio', fontsize=12, fontweight='bold')
        ax.set_ylabel('Recompensa Acumulada', fontsize=12, fontweight='bold')
        ax.set_title('Evolución del Aprendizaje - Recompensa por Episodio', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Anotaciones
        recompensa_final = np.mean(recompensas[-100:])
        ax.text(0.02, 0.98, 
               f'Recompensa promedio (últimos 100): {recompensa_final:.2f}',
               transform=ax.transAxes, 
               fontsize=10, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if guardar:
            ruta = os.path.join(self.carpeta_salida, 'evolucion_recompensas.png')
            plt.savefig(ruta, dpi=150, bbox_inches='tight')
            print(f"📊 Gráfica guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def graficar_pasos(self, pasos, ventana=50, guardar=True, mostrar=False):
        """
        Grafica la evolución del número de pasos por episodio
        
        Args:
            pasos: Lista de pasos por episodio
            ventana: Tamaño de ventana para promedio móvil
            guardar: Si guardar la gráfica
            mostrar: Si mostrar la gráfica
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodios = range(1, len(pasos) + 1)
        
        # Gráfica de pasos individuales
        ax.plot(episodios, pasos, alpha=0.3, color='green', 
                linewidth=0.5, label='Pasos por episodio')
        
        # Promedio móvil
        if len(pasos) >= ventana:
            promedio_movil = np.convolve(pasos, 
                                        np.ones(ventana)/ventana, 
                                        mode='valid')
            episodios_promedio = range(ventana, len(pasos) + 1)
            ax.plot(episodios_promedio, promedio_movil, 
                   color='darkgreen', linewidth=2, 
                   label=f'Promedio móvil (ventana={ventana})')
        
        # Configuración
        ax.set_xlabel('Episodio', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Pasos', fontsize=12, fontweight='bold')
        ax.set_title('Eficiencia del Agente - Pasos por Episodio', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Anotaciones
        pasos_final = np.mean(pasos[-100:])
        ax.text(0.02, 0.98, 
               f'Pasos promedio (últimos 100): {pasos_final:.1f}',
               transform=ax.transAxes, 
               fontsize=10, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        
        if guardar:
            ruta = os.path.join(self.carpeta_salida, 'evolucion_pasos.png')
            plt.savefig(ruta, dpi=150, bbox_inches='tight')
            print(f"📊 Gráfica guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def graficar_tasa_exito(self, exitos, ventana=50, guardar=True, mostrar=False):
        """
        Grafica la tasa de éxito a lo largo del entrenamiento
        
        Args:
            exitos: Lista de éxitos (1) y fallos (0) por episodio
            ventana: Tamaño de ventana para calcular tasa de éxito
            guardar: Si guardar la gráfica
            mostrar: Si mostrar la gráfica
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calcular tasa de éxito móvil
        tasas_exito = []
        episodios_tasa = []
        
        for i in range(ventana, len(exitos) + 1):
            tasa = np.mean(exitos[i-ventana:i]) * 100
            tasas_exito.append(tasa)
            episodios_tasa.append(i)
        
        # Gráfica
        ax.plot(episodios_tasa, tasas_exito, color='purple', linewidth=2)
        ax.fill_between(episodios_tasa, 0, tasas_exito, alpha=0.3, color='purple')
        
        # Línea de 100%
        ax.axhline(y=100, color='green', linestyle='--', linewidth=2, 
                  alpha=0.5, label='100% de éxito')
        
        # Configuración
        ax.set_xlabel('Episodio', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tasa de Éxito (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Tasa de Éxito del Agente (ventana={ventana} episodios)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Anotaciones
        tasa_final = np.mean(exitos[-100:]) * 100
        ax.text(0.02, 0.98, 
               f'Tasa de éxito final (últimos 100): {tasa_final:.1f}%',
               transform=ax.transAxes, 
               fontsize=10, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
        
        plt.tight_layout()
        
        if guardar:
            ruta = os.path.join(self.carpeta_salida, 'tasa_exito.png')
            plt.savefig(ruta, dpi=150, bbox_inches='tight')
            print(f"📊 Gráfica guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def graficar_trayectoria(self, entorno, trayectoria, guardar=True, mostrar=False):
        """
        Visualiza la trayectoria del agente en el grid
        
        Args:
            entorno: Objeto GridWorld
            trayectoria: Lista de estados visitados
            guardar: Si guardar la gráfica
            mostrar: Si mostrar la gráfica
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        size = entorno.size
        
        # Dibujar grid
        for i in range(size + 1):
            ax.plot([0, size], [i, i], 'k-', linewidth=0.5)
            ax.plot([i, i], [0, size], 'k-', linewidth=0.5)
        
        # Dibujar trampas
        for trampa in entorno.trampas:
            x, y = trampa
            rect = patches.Rectangle((y, size - x - 1), 1, 1, 
                                     linewidth=2, edgecolor='red', 
                                     facecolor='red', alpha=0.3)
            ax.add_patch(rect)
            ax.text(y + 0.5, size - x - 0.5, '💀', 
                   ha='center', va='center', fontsize=16)
        
        # Dibujar tesoro
        x, y = entorno.tesoro
        rect = patches.Rectangle((y, size - x - 1), 1, 1, 
                                 linewidth=2, edgecolor='gold', 
                                 facecolor='yellow', alpha=0.3)
        ax.add_patch(rect)
        ax.text(y + 0.5, size - x - 0.5, '💎', 
               ha='center', va='center', fontsize=16)
        
        # Dibujar inicio
        x, y = entorno.inicio
        rect = patches.Rectangle((y, size - x - 1), 1, 1, 
                                 linewidth=2, edgecolor='blue', 
                                 facecolor='lightblue', alpha=0.3)
        ax.add_patch(rect)
        ax.text(y + 0.5, size - x - 0.5, '🏁', 
               ha='center', va='center', fontsize=16)
        
        # Dibujar trayectoria
        if len(trayectoria) > 1:
            trayectoria_x = [size - estado[0] - 0.5 for estado in trayectoria]
            trayectoria_y = [estado[1] + 0.5 for estado in trayectoria]
            
            # Línea de trayectoria
            ax.plot(trayectoria_y, trayectoria_x, 
                   'b-', linewidth=3, alpha=0.6, label='Trayectoria')
            
            # Marcar puntos de la trayectoria
            ax.plot(trayectoria_y, trayectoria_x, 
                   'bo', markersize=8, alpha=0.4)
            
            # Numerar algunos puntos
            for i in [0, len(trayectoria)//2, -1]:
                if i < len(trayectoria):
                    ax.text(trayectoria_y[i], trayectoria_x[i], 
                           str(i+1), 
                           ha='center', va='center', 
                           fontsize=8, fontweight='bold',
                           color='white',
                           bbox=dict(boxstyle='circle', facecolor='blue', alpha=0.7))
        
        # Configuración
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.set_title(f'Trayectoria del Agente\nPasos totales: {len(trayectoria)-1}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Columna', fontsize=12)
        ax.set_ylabel('Fila', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if guardar:
            ruta = os.path.join(self.carpeta_salida, 'trayectoria_agente.png')
            plt.savefig(ruta, dpi=150, bbox_inches='tight')
            print(f"📊 Gráfica guardada: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def crear_dashboard(self, estadisticas, entorno, trayectoria, guardar=True, mostrar=False):
        """
        Crea un dashboard completo con todas las métricas
        
        Args:
            estadisticas: Diccionario con estadísticas del entrenamiento
            entorno: Objeto GridWorld
            trayectoria: Mejor trayectoria del agente
            guardar: Si guardar la gráfica
            mostrar: Si mostrar la gráfica
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Crear grid de subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Recompensas
        ax1 = fig.add_subplot(gs[0, :])
        recompensas = estadisticas['recompensas']
        episodios = range(1, len(recompensas) + 1)
        ax1.plot(episodios, recompensas, alpha=0.3, color='blue', linewidth=0.5)
        if len(recompensas) >= 50:
            promedio = np.convolve(recompensas, np.ones(50)/50, mode='valid')
            ax1.plot(range(50, len(recompensas) + 1), promedio, 
                    color='red', linewidth=2, label='Promedio móvil (50)')
        ax1.set_xlabel('Episodio', fontweight='bold')
        ax1.set_ylabel('Recompensa', fontweight='bold')
        ax1.set_title('Evolución de Recompensas', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Pasos
        ax2 = fig.add_subplot(gs[1, 0])
        pasos = estadisticas['pasos']
        episodios = range(1, len(pasos) + 1)
        ax2.plot(episodios, pasos, alpha=0.3, color='green', linewidth=0.5)
        if len(pasos) >= 50:
            promedio = np.convolve(pasos, np.ones(50)/50, mode='valid')
            ax2.plot(range(50, len(pasos) + 1), promedio, 
                    color='darkgreen', linewidth=2, label='Promedio móvil (50)')
        ax2.set_xlabel('Episodio', fontweight='bold')
        ax2.set_ylabel('Pasos', fontweight='bold')
        ax2.set_title('Número de Pasos por Episodio', fontweight='bold', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Tasa de éxito
        ax3 = fig.add_subplot(gs[1, 1])
        exitos = estadisticas['exitos']
        if len(exitos) >= 50:
            tasas = []
            eps = []
            for i in range(50, len(exitos) + 1):
                tasas.append(np.mean(exitos[i-50:i]) * 100)
                eps.append(i)
            ax3.plot(eps, tasas, color='purple', linewidth=2)
            ax3.fill_between(eps, 0, tasas, alpha=0.3, color='purple')
        ax3.axhline(y=100, color='green', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Episodio', fontweight='bold')
        ax3.set_ylabel('Tasa de Éxito (%)', fontweight='bold')
        ax3.set_title('Tasa de Éxito (ventana=50)', fontweight='bold', fontsize=12)
        ax3.set_ylim(-5, 105)
        ax3.grid(True, alpha=0.3)
        
        # 4. Trayectoria
        ax4 = fig.add_subplot(gs[2, :])
        size = entorno.size
        
        # Dibujar grid
        for i in range(size + 1):
            ax4.plot([0, size], [i, i], 'k-', linewidth=0.5)
            ax4.plot([i, i], [0, size], 'k-', linewidth=0.5)
        
        # Trampas, tesoro, inicio
        for trampa in entorno.trampas:
            x, y = trampa
            rect = patches.Rectangle((y, size - x - 1), 1, 1, 
                                     linewidth=1, edgecolor='red', 
                                     facecolor='red', alpha=0.3)
            ax4.add_patch(rect)
        
        x, y = entorno.tesoro
        rect = patches.Rectangle((y, size - x - 1), 1, 1, 
                                 linewidth=1, edgecolor='gold', 
                                 facecolor='yellow', alpha=0.3)
        ax4.add_patch(rect)
        
        x, y = entorno.inicio
        rect = patches.Rectangle((y, size - x - 1), 1, 1, 
                                 linewidth=1, edgecolor='blue', 
                                 facecolor='lightblue', alpha=0.3)
        ax4.add_patch(rect)
        
        # Trayectoria
        if len(trayectoria) > 1:
            tray_x = [size - estado[0] - 0.5 for estado in trayectoria]
            tray_y = [estado[1] + 0.5 for estado in trayectoria]
            ax4.plot(tray_y, tray_x, 'b-', linewidth=3, alpha=0.6)
            ax4.plot(tray_y, tray_x, 'bo', markersize=6, alpha=0.4)
        
        ax4.set_xlim(0, size)
        ax4.set_ylim(0, size)
        ax4.set_aspect('equal')
        ax4.set_title(f'Mejor Trayectoria del Agente (Pasos: {len(trayectoria)-1})', 
                     fontweight='bold', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Título general
        fig.suptitle('Dashboard de Entrenamiento del Agente Q-Learning', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if guardar:
            ruta = os.path.join(self.carpeta_salida, 'dashboard_completo.png')
            plt.savefig(ruta, dpi=150, bbox_inches='tight')
            print(f"📊 Dashboard guardado: {ruta}")
        
        if mostrar:
            plt.show()
        else:
            plt.close()
        
        return fig