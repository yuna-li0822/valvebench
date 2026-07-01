# METHODS

Describe how the system works: schema, pipeline, generator, models, and evaluation protocol.

Suggested sections:

- **Schema** — canonical `PATIENT_SCHEMA` and `SCHEMA` mapping (patient / scaffold fields)
- **Generator** — `generate_patient_dataset`, `generate_valve_dataset`, and dataset versioning
- **Splits** — group-preserving splitting (`create_group_splits`) and rationale
- **Baseline** — `BaselineMLP` configuration and training recipe
- **PINN** — `ForwardMLP`, training loop, and physics loss description
- **Evaluation** — metrics, OOD scenarios, statistical tests
