from schema.ast_schema import (
    ASTNode,
    ComparisonOp,
    LogicOp,
    MathLeaf,
    Constant
)

class ASTValidator:
    def __init__(self, allowed_variables: set):
        self.allowed_variables = allowed_variables
        self.errors = []

    def validate(self, node) -> bool:
        self.errors.clear()
        self._visit(node)
        return len(self.errors) == 0

    def _visit(self, node):
        if isinstance(node, ComparisonOp):
            self._visit(node.left)
            self._visit(node.right)

        elif isinstance(node, LogicOp):
            for op in node.operands:
                self._visit(op)

        elif isinstance(node, MathLeaf):
            self._validate_expression(node.expression)

        elif isinstance(node, Constant):
            pass  # always valid

        else:
            self.errors.append(f"Unknown AST node type: {type(node)}")

    def _validate_expression(self, expr: str):
        for token in expr.replace("*", " ").replace("+", " ").split():
            if "." in token:
                root = token.split(".", 1)[0]
                if root not in self.allowed_variables:
                    self.errors.append(
                        f"Illegal variable reference: {token}"
                    )
