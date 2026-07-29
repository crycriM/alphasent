## AGENTS.md guidelines

This repository hosts an alternative data pipeline for crypto sentiment. It is built to support both backtest mode and live mode. Two different sources of data are used: one for past data: GDELT and one for current data collating several RSS feeds, started accumulating.

### Detailed guidelines

The README.md file describes the CryptoPanic feed.
The PLAN.md document describes the whole project planning.

### Coding conventions

Modern Python using a project virtual env.

### LLM calls

Two self-hosted LLMs are used: llama3-8b (Q1-2024) and phi-4 (Q4-2024). It is served locally on port 8079 with an openAPI chat interface. They are purposely old models to avoid lookahead.

