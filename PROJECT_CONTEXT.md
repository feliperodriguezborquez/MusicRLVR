# MusicRLVR: Project Context & Manifesto

## El Problema
Los LLMs autoregresivos tradicionales componen música de izquierda a derecha (linealmente). Sin embargo, la composición musical humana es no-lineal: un compositor piensa en el clímax final y ajusta el principio para llegar ahí. Los LLMs "se pintan en una esquina" armónicamente.

## La Solución: TiDAR aplicado a Música
Implementamos una arquitectura inspirada en "Thought in Diffusion, Answer in Autoregression":
1.  **El "Pensamiento" (Difusión):** Un modelo de difusión genera una representación latente comprimida de la estructura completa (cadencias, modulaciones, contorno melódico). El modelo "ve" el futuro y el pasado simultáneamente durante el denosing.
2.  **La "Respuesta" (Autoregresión):** Un decodificador convierte esos latentes en tokens simbólicos (MIDI/ABC notation), asegurando la coherencia local.

## Metodología RLVR (Reinforcement Learning with Verifiable Rewards)
A diferencia del texto general, el contrapunto tiene reglas de "Verdad":
-   ¿Hay quintas paralelas? (True/False)
-   ¿La séptima resolvió bajando por grado conjunto? (True/False)
-   ¿Empezó y terminó en la tónica? (True/False)

Utilizaremos estas reglas programáticas como **Verificadores** en un bucle de entrenamiento tipo GRPO (o PPO modificado) para forzar al modelo a adherirse a la teoría musical perfecta.
