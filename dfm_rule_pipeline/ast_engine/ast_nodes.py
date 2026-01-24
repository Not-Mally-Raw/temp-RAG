from dataclasses import dataclass
from typing import List, Union, Literal, Optional

# Base Class
@dataclass
class ASTNode:
    pass

# 1. Logic Operations (Branching/Selection)
# Handles: max(), min(), if/else logic
@dataclass
class LogicOp(ASTNode):
    operator: Literal["max", "min", "and", "or"]
    operands: List[Union['ASTNode', 'MathLeaf', 'Constant']]

# 2. Comparison Operations (Pass/Fail Checks)
# Handles: A >= B, A < B
@dataclass
class ComparisonOp(ASTNode):
    operator: Literal[">", "<", ">=", "<=", "==", "!="]
    left: Union['ASTNode', 'MathLeaf', 'Constant']
    right: Union['ASTNode', 'MathLeaf', 'Constant']

# 3. Math Leaves (The safe python expressions at the bottom)
# Handles: "0.5 * ModuleParams.Thickness"
@dataclass
class MathLeaf(ASTNode):
    expression: str
    
    def __repr__(self):
        return f"Math('{self.expression}')"

# 4. Constants (Raw Numbers)
@dataclass
class Constant(ASTNode):
    value: float

    def __repr__(self):
        return f"Const({self.value})"