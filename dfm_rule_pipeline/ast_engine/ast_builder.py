import json
from .ast_nodes import ASTNode, LogicOp, ComparisonOp, MathLeaf, Constant

class ASTBuilder:
    @staticmethod
    def build(data: dict) -> ASTNode:
        """
        Recursively builds an AST from a dictionary.
        Expected format: 
        { "operator": "max", "operands": [...] } OR { "expression": "..." }
        """
        if not isinstance(data, dict):
            # If LLM passed a raw number or string, handle it
            if isinstance(data, (int, float)):
                return Constant(value=float(data))
            if isinstance(data, str):
                return MathLeaf(expression=data)
            raise ValueError(f"Unknown data type: {type(data)}")

        # CASE 1: Logic Operator (max, min)
        if "operator" in data and "operands" in data:
            op_type = data["operator"].lower()
            operands = [ASTBuilder.build(op) for op in data["operands"]]
            return LogicOp(operator=op_type, operands=operands)

        # CASE 2: Comparison (>=, <=)
        if "operator" in data and "left" in data and "right" in data:
            op_type = data["operator"]
            return ComparisonOp(
                operator=op_type,
                left=ASTBuilder.build(data["left"]),
                right=ASTBuilder.build(data["right"])
            )

        # CASE 3: Explicit Math Expression
        if "expression" in data:
            return MathLeaf(expression=data["expression"])

        # CASE 4: Explicit Constant
        if "value" in data:
            return Constant(value=float(data["value"]))

        raise ValueError(f"Invalid AST structure: {data.keys()}")

    @staticmethod
    def from_json(json_str: str) -> ASTNode:
        data = json.loads(json_str)
        return ASTBuilder.build(data)