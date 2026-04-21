# 🎵 MusicRLVR: Agentic Counterpoint Composer

¡Hola! Si estás leyendo esto, significa que has tomado las riendas de **MusicRLVR**. Este documento está diseñado para ponerte al día rápido sobre de qué trata el proyecto, qué hemos construido hasta ahora (mediados de 2026), hacia dónde va y cómo puedes continuar el trabajo.

## 🧠 ¿Cuál es el problema que resolvemos?
Los LLMs autoregresivos tradicionales (como los que generan texto) componen música linealmente, de izquierda a derecha. Sin embargo, la composición musical real, humana y estructurada (como el contrapunto de Bach) es inherentemente no-lineal. El compositor humano suele tener una "idea" o fin en mente (por ejemplo, una cadencia perfecta o modulación) y ajusta los compases previos para llegar allí coherentemente. Un LLM normal a menudo se "pinta en una esquina" armónicamente porque no puede ver el futuro.

## 🚀 Nuestra Solución: TiDAR + RLVR
Hemos propuesto una filosofía de Arquitectura y Entrenamiento para avanzar más allá del autoregresivo estándar:

1. **TiDAR (Thought in Diffusion, Answer in Autoregression):**
   - **Thought (Planner):** Un modelo de difusión genera una idea global y latente. Planifica cadencias, modulaciones y contornos enteros de forma no-lineal.
   - **Answer (Solver):** Un Transformer Autoregresivo toma este "plan latente" como contexto (vía Cross-Attention) y escribe las notas precisas token por token, asegurando coherencia local entre voces.
2. **RLVR (Reinforcement Learning with Verifiable Rewards):**
   - Para la música en estilo estricto (como el contrapunto de Fux), hay "verdades absolutas" deterministas (ej: evitar quintas u octavas paralelas).
   - En lugar de usar RLHF empírico, aplicamos **GRPO** (Group Relative Policy Optimization - como DeepSeek-R1) o PPO, utilizando validadores de teoría musical pura y dura como función de recompensa. Si el agente comete una "infracción" (quintas paralelas = -20 puntos), el LLM rápidamente reestructura sus pesos.

---

## 📂 ¿Qué hay en el repositorio ahora mismo? (Estado Actual)

Actualmente, tenemos construido el andamiaje del proyecto, el motor de reglas musicales estáticas y un pipeline de entrenamiento básico MVP interactuando.
*(Nota: Hemos reorganizado la raíz, moviendo los archivos gráficos antiguos generados a la carpeta `assets/` para facilitar la lectura del código).*

### Estructura de Carpetas

*   **`src/`**: ¡El corazón lógico!
    *   **`models/`**: Contiene la arquitectura Transformer autoregresiva (`policy.py`). Actualmente sirve como el "Actor" principal a entrenar.
    *   **`rl/`**: La magia del Reinforcement Learning. Incluye `grpo.py` y `ppo.py` para entrenar al actor, y `rewards.py` que contiene "El Examinador"; el script que impone Fuxian Species 1 (otorgando +1 por consonancia y -20 por quintas paralelas).
    *   **`theory/`**: Aquí se definen las estrictas reglas del contrapunto paramétricamente en `rules.yaml` (lee este archivo, es interesantísimo), así como generadores de cantus firmus iniciales para entrenar.
    *   **`data/`**: Scripts para lidiar con tensores de Pytorch (`dataset.py`) y para scrapear ejercicios reales de Bach mediante *music21* (`download_bach.py`).
    *   **`tools/`**: Herramientas auxiliares, por ejemplo, scripts de dibujado de partituras y visualizadores (`visualizer.py`) o para escuchar MIDI (`listen.py`).
*   **`generate_data.py`**: El script principal que genera un corpus sintético inicial de melodías aplicando reglas aleatorias dentro de parámetros legales para comenzar el proceso.
*   **`train_mvp.py`**: El entrypoint principal actual. Ensambla un bucle formativo con *GRPOEngine*. 
*   **`train_dummy.py`**: Script hiper simplista para asegurar que Pytorch y Cuda estén detectados correctamente en el PC local.
*   **`assets/`**: PNGs y SVGs con ejemplos visuales recientes de progresos de partitura.

Hay adicionalmente archivos markdown como `PROJECT_CONTEXT.md` y `architecture_spec.md`. Te recomiendo echarles un ojo pues ahí se condensaron especificaciones técnicas más granulares sobre los Transformers que pensábamos usar.

---

## 🛠️ ¿Cómo empezar a probar o jugar por tu cuenta?

### 1. Entorno y Dependencias
Principalmente requieres `torch` (con cuda recomendado para training), `music21` para la lógica en `src/theory` y `tensorboard`.
```bash
pip install torch music21 tensorboard
```
*Tip: Levanta el terminal y lanza `python train_dummy.py` para asegurar de que tu GPU es detectada.*

### 2. Generar el Dataset Sintético
Empieza por crear un corpus sintético de 5000 ejercicios base mediante reglas duras formales. En la carpeta del proyecto ejecuta:
```bash
python generate_data.py
```
*(Opcionalmente descargar datos de JS Bach puros: `python src/data/download_bach.py`)*

### 3. Entrenar al Agente (Lanzar MVP)
Una vez tengas datos (`data/synthetic_dataset.pt` o similares), arranca el bucle para ver cómo GRPO balancea parámetros, castiga reglas rotas y premia consonancias.
```bash
python train_mvp.py
```
Abre en paralelo:
```bash
tensorboard --logdir logs
```
*(No te preocupes si loss brinca un poco al inicio, los castigos por 5tas paralelas son muy agresivos).*

---

## 🚧 ¿Qué falta por hacer? (Next Steps)

Como te estás haciendo cargo, aquí te apuntamos qué desafíos y caminos quedaron inmediatamente pendientes tras las últimas discusiones grupales:

1. **Integración Completa de Difusión ("The Planner")**
   Tenemos robustecido el Solver (Transformer Causal en `src/models/` y Reglas en `rl/`), pero la red global de Difusión aún debe empalmarse inyectando su embedding en la fase pre-forward del autoregresivo. 
2. **Escalar las Recompensas (Rewards) y la Teoría**
   `rules.yaml` en `src/theory/` ya captura un gran andamiaje (rango max, tritonos prohibidos, saltos permitidos), pero `rewards.py` aún aplica esto empíricamente. Hay que integrar una función que calcule un puntaje ponderado 0-1 basándose estrictamente en si la generación satisfizo el yaml. Al momento, recompensa principalmente Especies 1 de forma base.
3. **Optimización de Gradientes en GRPO**
   Revisar si el castigo drástico de Fux (ej -20) no genera explosiones del gradiente u over-regularize (cayendo en Mode Collapse donde el agente compone solo un puñado de notas seguras por el resto de la vida). Ajustar la normalización de ventajas (Advantage).
4. **Callbacks Multimedia en Tensorboard**
   Aunque tenemos `music21` y `visualizer.py`, el loop de entrenamiento en `train_mvp.py` no exporta todavía las imágenes (SVGs o midis) autogeneradas a Tensorboard. Lograr esto mejoraría 1000% cómo se va evaluando la asimilación del sonido sin salir del dashboard.

¡Éxito con MusicRLVR! Estás a cargo de que la IA se convierta en el próximo Johann Joseph Fux del Siglo XXI.
