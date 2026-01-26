import json
import re
from llm.prompts import SELF_VALIDATE_PROMPT
from schema.feature_schema import features_dict


def get_schema_for_domain(domain):
    return features_dict.get(domain, features_dict.get("General", ""))


def self_validate(llm, formal_rule_str: str) -> dict:
    print("\n🔹 STAGE 4: SELF VALIDATION")

    try:
        formal = json.loads(formal_rule_str)
    except:
        return {"is_valid": False, "issues": ["Invalid formal rule JSON"]}

    # ✅ ACCEPT DEFERRALS
    if formal.get("status") == "Deferred":
        print("    ⏭️ Accepted (Deferred Rule)")
        return {"is_valid": True, "issues": []}


    domain = (
        formal.get("rule_json", {}).get("domain")
        or formal.get("domain", "General")
    )

    schema_context = get_schema_for_domain(domain)

    prompt = SELF_VALIDATE_PROMPT.format(
        formal_rule=json.dumps(formal, indent=2),
        schema_context=schema_context
    )

    raw = llm.call(prompt).strip()

    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "").strip()

    try:
        result = json.loads(raw)
    except:
        return {"is_valid": False, "issues": ["Validator JSON parse failure"]}

    # 🔒 FINAL SAFETY NET — illegal chained access
    equation = formal.get("equation")
    if equation:
        illegal = re.findall(r"\.(\w+)\.", equation)
        if illegal:
            result["is_valid"] = False
            result.setdefault("issues", []).append(
                f"Illegal chained attributes detected: {illegal}"
            )

    print("    ✅ Valid" if result["is_valid"] else f"    ❌ Invalid: {result['issues']}")
    return result
