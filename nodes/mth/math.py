import sys
import math
import numexpr as ne
import random
from ...core.node import node_path, node_prefix, main_cetegory

class MathInteger:
    
    CATEGORY = main_cetegory() + "/MTH"
    
    def __init__(cls):       
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "integer": ("INT", {"default": 0, "min": -sys.maxsize-1, "max": sys.maxsize}),              
            }
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("int", "number",)
    FUNCTION = "artha_main"
    
    @classmethod
    def artha_main(self, integer):
        
        return (integer, str(integer), )
        
#################################################
       
class MathFloat:
    
    CATEGORY = main_cetegory() + "/MTH"
    
    def __init__(cls):       
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0, "min": -sys.maxsize-1, "max": sys.maxsize, "step": 0.001, "round": 0.0001}),             
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("float", "number",)
    FUNCTION = "artha_main"
    
    @classmethod
    def artha_main(self, float):
        
        return (float, str(float), )
        
#################################################
         
class MathInt2Flo:
    
    CATEGORY = main_cetegory() + "/MTH"
    
    def __init__(cls):       
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "integer": ("INT", {"forceInput": True}),             
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("float", "number",)
    FUNCTION = "artha_main"
    
    @classmethod
    def artha_main(self, integer):
              
        return (float(integer), str(float(integer)), )

#################################################

class MathFlo2Int:
    
    CATEGORY = main_cetegory() + "/MTH"
    
    def __init__(cls):       
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"forceInput": True}),             
            }
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("integer", "number",)
    FUNCTION = "artha_main"
    
    @classmethod
    def artha_main(self, float):
              
        return (int(float), str(int(float)), )

################################################# 
    
class MathOperation:
    
    CATEGORY = main_cetegory() + "/MTH"
    
    class AnyType(str):
        def __ne__(self, __value: object) -> bool: return False
        
    any = AnyType("*")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (cls.any, {"forceInput": True}),
                "operation": ([
                    # binary
                    "add",
                    "subtract",
                    "multiply",
                    "divide",
                    "power",
                    "modulus",
                    "root",
                    "max",
                    "min",
                    "round",
                    "floor",
                    "ceil",
                    "clamp",
                    # unary
                    "abs",
                    "negate",
                    "sqrt",
                    "exp",
                    "log",
                    "sin",
                    "cos",
                    "tan"
                ], {"default": "add"}),
                "use_expression": ("BOOLEAN", {"default": False}),
                "expression": ("STRING", {"default": "a+b"}),
                "invert": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "b": (cls.any, {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    RETURN_NAMES = ("integer", "float", "number",)
    FUNCTION = "artha_main"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

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
    def artha_main(cls, a, b=None, use_expression=False, operation="add", expression="a+b", invert=False):

        a_f = cls.any_to_float(a)
        b_f = cls.any_to_float(b) if b is not None else None

        # invert swap (only if b exists)
        if invert and b_f is not None:
            a_f, b_f = b_f, a_f

        result = None

        # --- Expression Mode ---
        if use_expression:
            
            try:

                local_vars = {"a": a_f, "b": b_f if b_f is not None else 0.0}
                result = ne.evaluate(expression, local_dict=local_vars)
                result = float(result)
                
            except Exception:
                
                result = float("nan")

        # --- Normal Mode ---
        else:
            
            # Binary            
            if operation == "add":
                result = a_f + b_f if b_f is not None else float("nan")
            elif operation == "subtract":
                result = a_f - b_f if b_f is not None else float("nan")
            elif operation == "multiply":
                result = a_f * b_f if b_f is not None else float("nan")
            elif operation == "divide":
                result = a_f / b_f if b_f not in (0, None) else float("inf")
            elif operation == "power":
                result = a_f ** b_f if b_f is not None else float("nan")
            elif operation == "modulus":
                result = a_f % b_f if b_f not in (0, None) else float("nan")
            elif operation == "root":
                result = a_f ** (1 / b_f) if b_f not in (0, None) else float("nan")
            elif operation == "max":
                result = max(a_f, b_f) if b_f is not None else float("nan")
            elif operation == "min":
                result = min(a_f, b_f) if b_f is not None else float("nan")          
            # Unary
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
        
        result = cls.safe_round(result)
        
        # --- Outputs ---
        as_int = int(result) if result is not None and not math.isnan(result) else 0
        as_float = float(result) if result is not None else 0.0
        as_str = str(result) if result is not None else "NaN"

        return (as_int, as_float, as_str)

#################################################         
        
class MathRandom:
    
    CATEGORY = main_cetegory() + "/MTH"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "range_min": ("INT", {"default": 1, "min": 0, "max": sys.maxsize}),
                "range_max": ("INT", {"default": 100, "min": 0, "max": sys.maxsize}),                
            }
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("int", "number",)
    FUNCTION = "artha_main"
    
    def __init__(cls):
        
        pass
        
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        
        return (float("nan"), )

    @classmethod
    def artha_main(cls, range_min, range_max):
        
        low, high = min(range_min, range_max), max(range_min, range_max)
        
        integer = random.randint(low, high)
        
        return (integer, str(integer), )
        
#################################################
        
class MathNumber:
    
    CATEGORY = main_cetegory() + "/MTH"    
    DESCRIPTION = "Displays integer and float numbers as text."
    
    def __init__(self):
        
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number": ("STRING", {"forceInput": True}),
                "text": ("STRING", {
                    "default": ""
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("number",)
    FUNCTION = "artha_main"
    OUTPUT_NODE = True


    def artha_main(self, number, text):
       
        return {
        "ui": {"number": [number]}, 
        "result": (number,)
        }
        
#################################################



#################################################

NODE_CLASS_MAPPINGS = {
    "Math Integer":     MathInteger,
    "Math Float":       MathFloat,
    "Math Int2Flo":     MathInt2Flo,
    "Math Flo2Int":     MathFlo2Int,
    "Math Operation":   MathOperation,
    "Math Random":      MathRandom,
    "Math Number":      MathNumber,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Math Integer": node_prefix()   + " Math Integer",
    "Math Float": node_prefix()     + " Math Float",
    "Math Int2Flo": node_prefix()   + " Math I2F",
    "Math Flo2Int": node_prefix()   + " Math F2I",
    "Math Operation": node_prefix() + " Math Operation",
    "Math Random": node_prefix()    + " Math Random",
    "Math Number": node_prefix()    + " Math Number",
}