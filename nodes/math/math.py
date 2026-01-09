import sys
import math
import numexpr as ne
import random
from comfy_api.latest import io
from ...core.node import node_path, node_prefix, main_cetegory

class MathIntegerNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaMathInteger",
            display_name=node_prefix() + " Artha Math Integer",
            category=main_cetegory() + "/Math",
            inputs=[
                io.Int.Input(
                    "integer",
                    default=0,
                    min=-2**63,
                    max=2**63 - 1,
                ),
            ],
            outputs=[
                io.Int.Output("value"),
                io.String.Output("number"),
            ],
        )

    @classmethod
    def execute(cls, integer: int):
        return io.NodeOutput(
            integer,
            str(integer),
        )
        
#################################################
       
class MathFloatNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaMathFloat",
            display_name=node_prefix() + " Artha Math Float",
            category=main_cetegory() + "/Math",
            inputs=[
                io.Float.Input(
                    "float",
                    default=0.0,
                    min=-1e18,
                    max=1e18,
                    step=0.001,
                    round=0.0001,
                ),
            ],
            outputs=[
                io.Float.Output("value"),
                io.String.Output("number"),
            ],
        )

    @classmethod
    def execute(cls, float: float):
        return io.NodeOutput(
            float,
            str(float),
        )
        
#################################################
         
class MathI2FNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaMathI2F",
            display_name=node_prefix() + " Artha Math I2F",
            category=main_cetegory() + "/Math",
            inputs=[
                io.Int.Input(
                    "integer",
                    default=0,
                    force_input=True,
                ),
            ],
            outputs=[
                io.Float.Output("value", display_name="float",),
                io.String.Output("number"),
            ],
        )

    @classmethod
    def execute(cls, integer: int):
        value = float(integer)
        return io.NodeOutput(
            value,
            str(value),
        )

#################################################

class MathF2INode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaMathF2I",
            display_name=node_prefix() + " Artha Math F2I",
            category=main_cetegory() + "/Math",
            inputs=[
                io.Float.Input(
                    "float",
                    default=0.0,
                    force_input=True,
                ),
            ],
            outputs=[
                io.Int.Output("value", display_name="integer",),
                io.String.Output("number"),
            ],
        )

    @classmethod
    def execute(cls, float: float):
        value = int(float)
        return io.NodeOutput(
            value,
            str(value),
        )

################################################# 
    
class MathOperationNode(io.ComfyNode):

    OPERATIONS = [
        # binary
        "add", "subtract", "multiply", "divide", "power",
        "modulus", "root", "max", "min",
        # unary
        "round", "floor", "ceil",
        "abs", "negate", "sqrt", "exp", "log",
        "sin", "cos", "tan",
    ]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaMathOperation",
            display_name=node_prefix() + " Artha Math Operation",
            category=main_cetegory() + "/Math",
            inputs=[
                io.AnyType.Input(
                    "a",
                    display_name="a",
                ),
                io.AnyType.Input(
                    "b",
                    display_name="b",
                    optional=True,
                ),
                io.Combo.Input(
                    "operation",
                    options=cls.OPERATIONS,
                    default="add",
                ),
                io.Boolean.Input("use_expression", default=False),
                io.String.Input("expression", default="a+b"),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[
                io.Int.Output("integer"),
                io.Float.Output("float"),
                io.String.Output("number"),
            ],
        )

    @classmethod
    def any_to_float(cls, value):
        
        try:
            return float(value)
            
        except Exception:
            
            return 0.0

    @classmethod
    def safe_round(cls, value, digits=10):
        
        try:
            return round(float(value), digits)
            
        except Exception:
            
            return float("nan")

    @classmethod
    def execute(
        cls,
        a,
        operation="add",
        use_expression=False,
        expression="a+b",
        invert=False,
        b=None,
    ):
        a_f = cls.any_to_float(a)
        b_f = cls.any_to_float(b) if b is not None else None

        # invert (only if b exists)
        if invert and b_f is not None:
            a_f, b_f = b_f, a_f

        result = None

        # --- Expression mode ---
        if use_expression:
            try:
                local_vars = {
                    "a": a_f,
                    "b": b_f if b_f is not None else 0.0,
                }
                result = float(ne.evaluate(expression, local_dict=local_vars))
            except Exception:
                result = float("nan")

        # --- Normal mode ---
        else:
            try:
                if operation == "add":
                    result = a_f + b_f
                elif operation == "subtract":
                    result = a_f - b_f
                elif operation == "multiply":
                    result = a_f * b_f
                elif operation == "divide":
                    result = a_f / b_f if b_f not in (0, None) else float("inf")
                elif operation == "power":
                    result = a_f ** b_f
                elif operation == "modulus":
                    result = a_f % b_f if b_f not in (0, None) else float("nan")
                elif operation == "root":
                    result = a_f ** (1 / b_f) if b_f not in (0, None) else float("nan")
                elif operation == "max":
                    result = max(a_f, b_f)
                elif operation == "min":
                    result = min(a_f, b_f)

                # unary
                elif operation == "round":
                    result = round(a_f)
                elif operation == "floor":
                    result = math.floor(a_f)
                elif operation == "ceil":
                    result = math.ceil(a_f)
                elif operation == "abs":
                    result = abs(a_f)
                elif operation == "negate":
                    result = -a_f
                elif operation == "sqrt":
                    result = math.sqrt(a_f) if a_f >= 0 else float("nan")
                elif operation == "exp":
                    result = math.exp(a_f)
                elif operation == "log":
                    result = math.log(a_f) if a_f > 0 else float("nan")
                elif operation == "sin":
                    result = math.sin(a_f)
                elif operation == "cos":
                    result = math.cos(a_f)
                elif operation == "tan":
                    result = math.tan(a_f)

            except Exception:
                result = float("nan")

        result = cls.safe_round(result)

        as_int = int(result) if result is not None and not math.isnan(result) else 0
        as_float = float(result) if result is not None else 0.0
        as_str = str(result) if result is not None else "NaN"

        return io.NodeOutput(
            as_int,
            as_float,
            as_str,
        )
        
#################################################         
        
class MathRandomNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaMathRandom",
            display_name=node_prefix() + " Artha Math Random",
            category=main_cetegory() + "/Math",
            inputs=[
                io.Int.Input(
                    "range_min",
                    default=1,
                    min=0,
                    max=2**63 - 1,
                ),
                io.Int.Input(
                    "range_max",
                    default=100,
                    min=0,
                    max=2**63 - 1,
                ),
            ],
            outputs=[
                io.Int.Output("value"),
                io.String.Output("number"),
            ],
        )

    @classmethod
    def execute(cls, range_min: int, range_max: int):
        low, high = min(range_min, range_max), max(range_min, range_max)
        value = random.randint(low, high)
        return io.NodeOutput(
            value,
            str(value),
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Always return a changing value → force re-execution
        return float("nan")
        
#################################################
        
class MathNumberNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaMathNumber",
            display_name=node_prefix() + " Artha Math Number",
            category=main_cetegory() + "/Math",
            description="Displays integer and float numbers as text.",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "number",
                    force_input=True,
                ),
                io.String.Input(
                    "text",
                    default="",
                ),
            ],
            outputs=[
                io.String.Output(
                "out_number",
                display_name="number",
                ),
            ],
        )

    @classmethod
    def execute(cls, number: str, text: str):
        
        print(f"{node_prefix()} [ArthaMathNumberNode] Output : {number}")

        ui = {
            "number": [number],
        }

        return io.NodeOutput(
            number,
            ui=ui,
        )
        
#################################################