# WS-OVVIS Scientific Overlay Status (v2 / v10-refined)

This file tracks the current authoritative state for the scientific validation overlay.

## Schema
- Scientific overlay version
- Formal active gate
- Diagnostic active gate
- Scientific mode (`formal` / `diagnostic` / `recovery` / `migration`)
- Last formal PASS
- Last formal FAIL
- Current recovery gate
- Later formal gate activation allowed
- Diagnostic-only probing allowed
- Human sign-off status
- Notes on historical evidence carried over from prior overlay versions

## Current migration state
- Scientific overlay version: `v2 / v10-refined`
- Scientific mode: `migration`
- Formal active gate: `P0`
- Diagnostic active gate: `none`
- Last formal PASS: `none under refined overlay yet`
- Last formal FAIL: `historical strict S1 FAIL retained as legacy evidence only`
- Current recovery gate: `none`
- Later formal gate activation allowed: `no`
- Diagnostic-only probing allowed: `only if explicitly approved by the refined charter`
- Human sign-off status: `refined overlay migration pending`

## Notes
- This file supersedes the simpler prior scientific STATUS schema.
- Historical `S1 FAIL` / `S1R` materials remain preserved as legacy evidence, but do not define refined-overlay progression.
