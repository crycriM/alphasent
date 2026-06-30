## AGENTS.md guidelines

This repository hosts an alternative data pipeline for crypto sentiment. It is built to support both backtest mode and live mode. Two different sources of data are used: one for past data: GDELT and one for current data, CryptoPanic, that will start accumulating when ready.

### Detailed guidelines

The README.md file describes the CryptoPanic feed.
The PLAN.md document describes the whole project planning.

### Coding conventions

Modern Python using a project virtual env.

### LLM calls

A self-hosted LLM will be used: llama3-8b. It currently runs on this machine on port 8079 with an openAPI chat interface. Its context is small (8k) and it's purposely an old model to avoid lookahead.


