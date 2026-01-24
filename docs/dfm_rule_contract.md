# DFM Rule Contract (Conceptual)

This is the non-code contract all components must obey.

## Rule (atomic)
- intent: single, atomic manufacturing invariant
- scope_domain: {process, item, feature}
- applicability (hard gates): list of conditions such as material/process/feature presence; outside gates the rule must not evaluate
- constraints[]: evaluatable expressions; each with subject, operator, value, unit, logic (simple|compound), provenance_quote
- severity: ENFORCEABLE | ADVISORY
- validation_state: ENFORCEABLE | ADVISORY_ONLY | INCOMPLETE | AMBIGUOUS (set by extraction/validator)
- provenance: document, section/page, chunk, supporting_quotes

## Constraints
- Simple: subject + operator (<, <=, >=, ==, !=, between, ±) + value (+ unit)
- Compound: operator MAX/MIN applied to operands (literals or expressions); must be structured, not prose
- Every enforceable rule must have ≥1 constraint; advisory rules may have none

## Applicability
- Binary gating logic (material/process/feature flags)
- Must be captured explicitly (e.g., material == "low carbon steel"; feature == "bend")

## Advisory vs Enforceable
- Advisory: heuristics like "avoid", "for ease of manufacturing"
- Enforceable: deterministic constraints ("shall", "must", min/max, tolerance")

## Validation expectations
- ENFORCEABLE only when evaluatable constraints exist
- ADVISORY_ONLY when severity=ADVISORY
- INCOMPLETE when constraints missing/ill-formed
- AMBIGUOUS reserved for conflicts/uncertain parsing

## Golden Acceptance Test (Sheet Metal Bends)
Input snippet:
```
Bends
Bends should be toleranced plus or minus one-half degree at a location adjacent to the bends.
For the ease of manufacturing, multiple bends on the same plane should occur in the same direction.
Avoid large sheet metal parts with small bent flanges.
In low carbon steel sheet metal, the minimum radius of a bend should be one-half the material thickness
or 0.80 mm (0.03 inch), whichever is larger.
```

Expected rules (conceptual):
1) Bend angle tolerance — ENFORCEABLE
   - applicability: feature == "bend"
   - constraint: bend_angle, operator: ±, value: 0.5, unit: degree
2) Bend direction consistency — ADVISORY
   - applicability: multiple_bends==true, same_plane==true
   - constraint: bend_direction == same_direction
3) Flange size guideline — ADVISORY
   - applicability: feature == "flange"
   - constraint: flange_size operator: avoid value: small_on_large_part
   - should not reach validator as enforceable
4) Minimum bend radius — ENFORCEABLE
   - applicability: material == "low carbon steel"; feature == "bend"
   - constraint: bend_radius >= MAX(0.5*material_thickness, 0.80 mm)
