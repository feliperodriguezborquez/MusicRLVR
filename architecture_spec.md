# Architecture Specification

## 1. The Planner (Diffusion Model)
- **Input:** Prompt de texto (ej: "Contrapunto a 2 voces, modo Dórico, 16 compases").
- **Architecture:** DiT (Diffusion Transformer) pequeño o U-Net 1D.
- **Output:** `PlanLatent` (Tensor [Batch, Seq_Len/Compression, Hidden_Dim]). Representa la "intención" armónica global.

## 2. The Bridge (Adapter)
- Una capa de proyección (MLP o Cross-Attention) que inyecta el `PlanLatent` en el espacio de embedding del LLM.

## 3. The Solver (Autoregressive LLM)
- **Base:** Un modelo pequeño (ej: Llama-3-Nano, Qwen-2.5-Coder-0.5B, o similar disponible en 2026).
- **Mechanism:** Standard Causal Masking, pero con Cross-Attention atendiendo al `PlanLatent`.
- **Output:** Tokens de música.

## 4. The Verifier (RL Environment)
- **Input:** Secuencia de tokens generada.
- **Engine:** `music21` script.
- **Rules:**
    - `no_parallel_fifths(score) -> float`
    - `no_parallel_octaves(score) -> float`
    - `check_stepwise_motion(score) -> float`
    - `check_cadence(score) -> float`
- **Reward:** Suma ponderada de reglas pasadas (Binario 1.0 si pasa todo, o penalización fraccionada).
