---
title: GuardRails Architecture
---
flowchart LR
    A[User Request] -->B(Input Guaredrails \n - Moderation \n - Injection scan)
    B --> C>LLM]
    C -->D[Output Guardrails \n - Schema validate \n - Citation checks \n - refusal checks]
    D -->E[Final API]
