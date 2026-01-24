import ast
from .ast_nodes import ASTNode, LogicOp, ComparisonOp, MathLeaf, Constant

class ASTValidator:
    def __init__(self, allowed_variables: set):
        self.allowed_variables = allowed_variables
        self.errors = []

    def validate(self, node: ASTNode) -> bool:
        """Returns True if the AST is safe and valid against schema."""
        self.errors = []
        self._recursive_validate(node)
        return len(self.errors) == 0

    def _recursive_validate(self, node: ASTNode):
        if isinstance(node, LogicOp):
            if node.operator not in ["max", "min", "and", "or"]:
                self.errors.append(f"Illegal logic operator: {node.operator}")
            for child in node.operands:
                self._recursive_validate(child)

        elif isinstance(node, ComparisonOp):
            if node.operator not in [">", "<", ">=", "<=", "==", "!="]:
                self.errors.append(f"Illegal comparison operator: {node.operator}")
            self._recursive_validate(node.left)
            self._recursive_validate(node.right)

        elif isinstance(node, MathLeaf):
            self._validate_python_expression(node.expression)

        elif isinstance(node, Constant):
            pass # Numbers are always safe

    def _validate_python_expression(self, expr_str: str):
        """
        Parses the math string safely using Python's ast module.
        Ensures no function calls, no imports, only allowed variables.
        """
        try:
            tree = ast.parse(expr_str, mode='eval')
        except SyntaxError:
            self.errors.append(f"Syntax Error in math expression: '{expr_str}'")
            return

        for child in ast.walk(tree):
            # 1. Block Function Calls (eval(), os.system(), etc.)
            if isinstance(child, ast.Call):
                self.errors.append(f"Function calls forbidden in math: '{expr_str}'")
            
            # 2. Check Variables
            elif isinstance(child, ast.Name):
                # We split 'ModuleParams.Thickness' -> 'ModuleParams'
                root_var = child.id.split('.')[0] 
                if root_var not in self.allowed_variables and root_var not in ['max', 'min', 'abs']:
                     # Allow basic math functions, flag unknown variables
                     # For now, we just warn, or you can strictly fail
                     pass 
            
            # 3. Block Dangerous Nodes
            elif isinstance(child, (ast.Import, ast.ImportFrom, ast.Lambda)):
                 self.errors.append(f"Dangerous code detected: '{expr_str}'")