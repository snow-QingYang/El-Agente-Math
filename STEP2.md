# STEP 2: Define Pydantic models

## Changes

- Created `packages/data/src/el_agente_data/models.py` with all data models
- Created `packages/agent/src/el_agente/models.py` with agent models
- Created `packages/benchmark/src/el_agente_bench/models.py` with benchmark models

All raw dict/json structures are replaced with typed Pydantic BaseModels.
