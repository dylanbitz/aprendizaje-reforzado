from flask import Flask, render_template, request, jsonify
from flask_sock import Sock
import json
import time
from entorno import GridWorld
from agente import AgenteQLearning

app = Flask(__name__)
sock = Sock(app)

# Variables globales
entorno_global = None
agente_global = None
entrenamiento_activo = False

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/agente')
def agente():
    """Página del agente"""
    return render_template('agente.html')

@app.route('/api/configurar', methods=['POST'])
def configurar():
    """Configura el entorno y el agente"""
    global entorno_global, agente_global

    try:
        data = request.json
        size = int(data.get('size', 10))
        num_trampas = int(data.get('num_trampas', 10))
        max_pasos = int(data.get('max_pasos', 200))
        alpha = float(data.get('alpha', 0.1))
        gamma = float(data.get('gamma', 0.9))
        epsilon = float(data.get('epsilon', 0.1))

        # Crear entorno
        entorno_global = GridWorld(size=size, num_trampas=num_trampas, max_pasos=max_pasos)

        # Crear agente
        agente_global = AgenteQLearning(
            num_estados=entorno_global.get_num_estados(),
            num_acciones=entorno_global.get_num_acciones(),
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )

        return jsonify({
            'status': 'success',
            'entorno': entorno_global.get_estado_grid()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'mensaje': str(e)}), 400

@sock.route('/ws/entrenar')
def entrenar_ws(ws):
    """WebSocket para entrenamiento en tiempo real"""
    global entorno_global, agente_global, entrenamiento_activo

    entrenamiento_activo = True

    try:
        # Recibir configuración inicial
        config_msg = ws.receive()
        config = json.loads(config_msg)

        num_episodios = int(config.get('num_episodios', 100))
        velocidad = float(config.get('velocidad', 0.01))  # Segundos de delay

        victorias = 0

        for episodio in range(num_episodios):
            if not entrenamiento_activo:
                break

            # Reset entorno
            estado = entorno_global.reset()
            estado_indice = entorno_global.estado_a_indice(estado)

            pasos_episodio = 0
            exito = False

            # Ejecutar episodio
            while True:
                # Elegir acción
                accion = agente_global.elegir_accion(estado_indice, entrenar=True)

                # Ejecutar acción
                siguiente_estado, recompensa, terminado, info = entorno_global.step(accion)
                siguiente_estado_indice = entorno_global.estado_a_indice(siguiente_estado)

                # Actualizar Q
                agente_global.actualizar_q(estado_indice, accion, recompensa,
                                          siguiente_estado_indice, terminado)

                pasos_episodio = info['pasos']
                estado_indice = siguiente_estado_indice

                # Enviar estado actual al cliente
                ws.send(json.dumps({
                    'tipo': 'paso',
                    'grid': entorno_global.get_estado_grid(),
                    'episodio': episodio + 1,
                    'pasos': pasos_episodio
                }))

                time.sleep(velocidad)

                if terminado:
                    exito = info.get('exito', False)
                    if exito:
                        victorias += 1
                    break

            # Enviar resumen del episodio
            ws.send(json.dumps({
                'tipo': 'episodio_completo',
                'episodio': episodio + 1,
                'pasos': pasos_episodio,
                'exito': exito,
                'victorias': victorias,
                'tasa_exito': (victorias / (episodio + 1)) * 100
            }))

        # Entrenamiento completado
        ws.send(json.dumps({
            'tipo': 'entrenamiento_completo',
            'episodios_totales': num_episodios,
            'victorias': victorias,
            'tasa_exito_final': (victorias / num_episodios) * 100
        }))

    except Exception as e:
        ws.send(json.dumps({
            'tipo': 'error',
            'mensaje': str(e)
        }))
    finally:
        entrenamiento_activo = False

@app.route('/api/detener', methods=['POST'])
def detener():
    """Detiene el entrenamiento"""
    global entrenamiento_activo
    entrenamiento_activo = False
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
    