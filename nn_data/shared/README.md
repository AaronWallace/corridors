# Shared training datasets

Datasets placed here are intentionally visible to Git and remain available in
Corridors training-data selectors. Use **Dataset Manager → `s #`** to move an
active dataset here while preserving its logical type and metadata.

For example, an AlphaZero run named `alphazero/my_run` becomes
`shared/alphazero/my_run`.

The machine-local `.dataset-index.json` cache is intentionally ignored. Each
system recreates that lightweight index without modifying the committed data.
