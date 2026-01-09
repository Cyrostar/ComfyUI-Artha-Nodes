from comfy_api.latest import io
from ...core.node import node_path, node_prefix, main_cetegory

class LogicBasicNode(io.ComfyNode):
    
    STATES = ["true", "false",]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaLogicBasic",
            display_name=node_prefix() + " Artha Logic Basic",
            category=main_cetegory() + "/Logic",
            inputs=[
                io.Combo.Input(
                    "state",
                    options=cls.STATES,
                    default="true",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    "out_state",
                    display_name="state",
                ),
            ],
        )

    @classmethod
    def execute(cls, state: str) -> tuple[bool]:
        
        state = state.lower() == "true"

        return (state,)
     
#################################################

class LogicGateNode(io.ComfyNode):

    GATES = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "NOT",]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaLogicGate",
            display_name=node_prefix() + " Artha Logic Gate",
            category=main_cetegory() + "/Logic",
            inputs=[
                io.Combo.Input(
                    "gate",
                    options=cls.GATES,
                    default="AND",
                ),
                io.Boolean.Input("a", display_name="a"),
                io.Boolean.Input(
                    "b",
                    display_name="b",
                    optional=True,
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    "out_state", 
                    display_name="state"
                ),
            ],
        )

    @classmethod
    def execute(cls, gate: str, a: bool, b: bool = False) -> tuple[bool]:
        
        gate = gate.upper()

        if gate == "AND":
            out = a and b
        elif gate == "OR":
            out = a or b
        elif gate == "XOR":
            out = a != b
        elif gate == "NAND":
            out = not (a and b)
        elif gate == "NOR":
            out = not (a or b)
        elif gate == "XNOR":
            out = a == b
        elif gate == "NOT":
            out = not a
        else:
            raise ValueError(f"Unknown gate type: {gate}")

        return (out,)
        
#################################################

class LogicCompareNode(io.ComfyNode):

    OPERATORS = ["==", "!=", ">", "<", ">=", "<=",]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaLogicCompare",
            display_name=node_prefix() + " Artha Logic Compare",
            category=main_cetegory() + "/Logic",
            inputs=[
                io.Combo.Input(
                    "op",
                    options=cls.OPERATORS,
                    default="==",
                ),
                io.AnyType.Input(
                    "a",
                    display_name="a",
                ),
                io.AnyType.Input(
                    "b",
                    display_name="b",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    "out_state",
                    display_name="state",
                ),
            ],
        )

    @classmethod
    def execute(cls, op: str, a, b) -> tuple[bool]:
        if op == "==":
            out = a == b
        elif op == "!=":
            out = a != b
        elif op == ">":
            out = a > b
        elif op == "<":
            out = a < b
        elif op == ">=":
            out = a >= b
        elif op == "<=":
            out = a <= b
        else:
            raise ValueError(f"Unknown operator: {op}")

        return (bool(out),)

#################################################

class LogicPassthruNode(io.ComfyNode):
    
    STATES = ["true", "false",]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaLogicPassthru",
            display_name=node_prefix() + " Artha Logic Passthru",
            category=main_cetegory() + "/Logic",
            inputs=[
                io.AnyType.Input(
                    "any",
                    display_name="any",
                ),
                io.Combo.Input(
                    "state",
                    options=cls.STATES,
                    default="true",
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    "out_any",
                    display_name="any",
                ),
                io.Boolean.Output(
                    "out_state",
                    display_name="state",
                ),
            ],
        )

    @classmethod
    def execute(cls, any, state: str):
        
        state = state.lower() == "true"
        
        if state:
            
            return (any, state,)
            
        else:
            
            return (None, state,)
            
#################################################
       
class LogicPrintNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaLogicPrint",
            display_name=node_prefix() + " Artha Logic Print",
            category=main_cetegory() + "/Logic",
            is_output_node=True,
            inputs=[
                io.Boolean.Input(
                    "state",
                    display_name="state",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    "out_state",
                    display_name="state",
                ),
            ],
        )

    @classmethod
    def execute(cls, state: bool) -> tuple[bool]:
        
        print(f"[Artha Logic Print] State = {state}")
        return (state,)
        
        
        
