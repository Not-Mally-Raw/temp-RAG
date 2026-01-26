def requires_ast(rule_text: str) -> bool:
    triggers = [
        "whichever is larger",
        "whichever is smaller",
        "greater of",
        "lesser of",
        "maximum of",
        "minimum of"
    ]
    return any(t in rule_text.lower() for t in triggers)
