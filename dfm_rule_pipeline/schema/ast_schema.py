# ast_schema.py

from dataclasses import dataclass
from typing import List, Union

class ASTNode:
    pass


@dataclass
class Constant(ASTNode):
    value: float


@dataclass
class MathLeaf(ASTNode):
    """
    Symbolic math expression evaluated against context.
    Example: "0.5 * ModuleParams.Thickness"
    """
    expression: str


@dataclass
class LogicOp(ASTNode):
    """
    max, min, and, or
    """
    operator: str
    operands: List[ASTNode]


@dataclass
class ComparisonOp(ASTNode):
    """
    >=, <=, >, <, ==
    """
    operator: str
    left: ASTNode
    right: ASTNode

from dataclasses import dataclass
from typing import List, Union

# -------------------------------------------------------------------
# AST NODE DEFINITIONS
# -------------------------------------------------------------------

class ASTNode:
    pass

@dataclass
class Constant(ASTNode):
    value: float

@dataclass
class MathLeaf(ASTNode):
    """
    Symbolic math expression evaluated against context.
    Example: "0.5 * ModuleParams.Thickness"
    """
    expression: str

@dataclass
class LogicOp(ASTNode):
    """
    max, min, and, or
    """
    operator: str
    operands: List[ASTNode]

@dataclass
class ComparisonOp(ASTNode):
    """
    >=, <=, >, <, ==
    """
    operator: str
    left: ASTNode
    right: ASTNode


# -------------------------------------------------------------------
# SERIALIZER (MISSING FUNCTION)
# -------------------------------------------------------------------

def serialize_ast(node):
    if isinstance(node, ComparisonOp):
        return {
            "type": "ComparisonOp",
            "operator": node.operator,
            "left": serialize_ast(node.left),
            "right": serialize_ast(node.right),
        }

    if isinstance(node, LogicOp):
        return {
            "type": "LogicOp",
            "operator": node.operator,
            "operands": [serialize_ast(o) for o in node.operands],
        }

    if isinstance(node, MathLeaf):
        return {
            "type": "MathLeaf",
            "expression": node.expression,
        }

    if isinstance(node, Constant):
        return {
            "type": "Constant",
            "value": node.value,
        }

    raise TypeError(f"Unknown AST node type: {type(node)}")