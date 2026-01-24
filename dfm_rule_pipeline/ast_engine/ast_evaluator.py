from .ast_nodes import ASTNode, LogicOp, ComparisonOp, MathLeaf, Constant

class ASTEvaluator:
    def __init__(self, context: dict):
        self.context = context # e.g. {'ModuleParams': {'Thickness': 1.2}}

    def evaluate(self, node: ASTNode):
        if isinstance(node, Constant):
            return node.value

        if isinstance(node, MathLeaf):
            # Safe evaluation using the context
            # NOTE: In production, use a library like 'simpleeval' instead of eval()
            return eval(node.expression, {"__builtins__": None}, self.context)

        if isinstance(node, LogicOp):
            values = [self.evaluate(op) for op in node.operands]
            if node.operator == "max":
                return max(values)
            elif node.operator == "min":
                return min(values)
            elif node.operator == "and":
                return all(values)
            elif node.operator == "or":
                return any(values)

        if isinstance(node, ComparisonOp):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            
            if node.operator == ">": return left > right
            if node.operator == "<": return left < right
            if node.operator == ">=": return left >= right
            if node.operator == "<=": return left <= right
            if node.operator == "==": return left == right
            
        raise ValueError(f"Cannot evaluate node: {node}")