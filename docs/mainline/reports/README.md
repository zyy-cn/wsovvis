# WS-OVVIS Mainline Reports

This directory stores the durable outputs required by the document-driven automation loop.

Expected durable files:
- `phase_gate_latest.txt`
- `acceptance_latest.txt`
- `evidence_latest.txt`
- `worked_example_verification_latest.md`
- `worked_example_verification_latest.json`
- `supervisor_prompt_latest.txt` when generated
- `mainline_terminal_summary.txt` when terminal mode is active

Archive policy:
- every bounded loop should also write timestamped copies under `archive/`
- do not rely only on chat memory for gate justification
