import json
from llm.prompts import SCHEMA_PROMPT
from schema.feature_schema import features_dict


def validate_against_schema(llm, intent_input) -> str:
    """
    Schema mapping with explicit deferral semantics.
    This function MUST NOT hallucinate schema mappings.
    """
    print("\n🔹 STAGE 2: SCHEMA MAPPING")

    # ------------------------------------------------------------------
    # 1. Normalize Input
    # ------------------------------------------------------------------
    if isinstance(intent_input, dict):
        intent = intent_input
    else:
        try:
            intent = json.loads(intent_input)
        except Exception:
            return json.dumps({
                "schema_valid": False,
                "error_type": "InvalidInput",
                "error": "Intent JSON is invalid"
            })

    # ------------------------------------------------------------------
    # 2. HARD DEFERRALS (INTENT-LEVEL, NOT FAILURES)
    # ------------------------------------------------------------------
    if intent.get("requires_geometry") is True:
        return json.dumps({
            **intent,
            "schema_valid": False,
            "error_type": "DeferredGeometry",
            "error": "Deferred: Geometry Layer"
        })

    if intent.get("requires_tolerance") is True:
        return json.dumps({
            **intent,
            "schema_valid": False,
            "error_type": "DeferredTolerance",
            "error": "Deferred: Tolerance Spec"
        })

    if intent.get("is_quantifiable") is False:
        return json.dumps({
            **intent,
            "schema_valid": False,
            "error_type": "AdvisoryRule",
            "error": "Non-quantifiable advisory rule"
        })

    # ------------------------------------------------------------------
    # 3. LLM SCHEMA MAPPING
    # ------------------------------------------------------------------
    domain = intent.get("domain", "General")
    schema_text = features_dict.get(domain, features_dict.get("General", ""))

    prompt = SCHEMA_PROMPT.format(
        schema=schema_text,
        intent_json=json.dumps(intent, indent=2)
    )

    try:
        raw = llm.call(prompt)
        if not isinstance(raw, str):
            raw = str(raw)

        raw = raw.strip()

        if "```" in raw:
            for part in raw.split("```"):
                p = part.strip()
                if p.startswith("{") and p.endswith("}"):
                    raw = p
                    break

        print(f"    → Schema Mapping (raw): {raw[:400]}")

    except Exception as e:
        return json.dumps({
            "schema_valid": False,
            "error_type": "LLMFailure",
            "error": str(e)
        })

    # ------------------------------------------------------------------
    # 4. STRICT PARSE + CLASSIFICATION
    # ------------------------------------------------------------------
    try:
        parsed = json.loads(raw)
    except Exception as e:
        return json.dumps({
            "schema_valid": False,
            "error_type": "SchemaParseError",
            "error": str(e)
        })

    # ------------------------------------------------------------------
    # 5. SUPPORTED VS UNDERSTOOD
    # ------------------------------------------------------------------
    if parsed.get("schema_valid") is True:
        if not parsed.get("object") or not parsed.get("attribute"):
            return json.dumps({
                "schema_valid": False,
                "error_type": "IncompleteSchemaMatch",
                "error": "Schema mapping incomplete despite schema_valid true"
            })
        return raw

    # Explicitly recognized but unsupported
    if "Surface" in (parsed.get("object") or ""):
        return json.dumps({
            **intent,
            "schema_valid": False,
            "error_type": "UnsupportedPMI",
            "error": "PMI feature recognized but not supported by schema"
        })

    if "Shaft" in (parsed.get("object") or ""):
        return json.dumps({
            **intent,
            "schema_valid": False,
            "error_type": "UnsupportedFeature",
            "error": "Feature type 'Shaft' recognized but not supported"
        })

    return raw
