from ...core.node import node_path, node_prefix, main_cetegory

class InputSelector:
    
    CATEGORY = main_cetegory() + "/WFO"
    
    class AnyType(str):
        def __ne__(self, __value: object) -> bool: return False
        
    any = AnyType("*")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["SELECT A AS OUTPUT", "SELECT B AS OUTPUT", "NOT NULL AS OUTPUT", "BOTH TRUE A AS OUTPUT", "BOTH TRUE B AS OUTPUT"], {
                    "default": "SELECT A AS OUTPUT"
                }),
            },
            "optional": {
                "val_a": (cls.any, {"forceInput": True}),
                "val_b": (cls.any, {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("O",)
    FUNCTION = "artha_main"
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def artha_main(cls, operation, val_a=None, val_b=None):

        if operation == "SELECT A AS OUTPUT":
            
            return (val_a,)
            
        elif operation == "SELECT B AS OUTPUT":
            
            return (val_b,)
            
        elif operation == "NOT NULL AS OUTPUT":
 
            if cls.checker(val_a) and not cls.checker(val_b):
                
                return (val_a,)
                
            elif cls.checker(val_b) and not cls.checker(val_a):
                
                return (val_b,)
                
            if cls.checker(val_a) and cls.checker(val_b):
                
                return (val_a,)
            
            else:
                
                raise ValueError("A and B not valid.")
                
        elif operation == "BOTH TRUE A AS OUTPUT":
            
            if cls.checker(val_a) and cls.checker(val_b):
                
                return (val_a,)
                
        elif operation == "BOTH TRUE B AS OUTPUT":
            
            if cls.checker(val_a) and cls.checker(val_b):
                
                return (val_b,)
                
        else:
                return (A,)
   
    @classmethod
    def checker(cls, value):

        if value is None:
            
            return False

        if isinstance(value, bool):
            
            return value

        if isinstance(value, (int, float)):
            
            return value != 0
            
        if isinstance(value, str):
            
            return len(value) > 0 and value.lower() not in ['false', '0', 'none', 'null', '']

        if isinstance(value, list):
            
            return len(value) > 0           

        if isinstance(value, dict):
            
            return len(value) > 0           

        if isinstance(value, tuple):
            
            return len(value) > 0
            
        if isinstance(value, set):
            
            return len(value) > 0
            
        if hasattr(value, '__len__'):
            
            try:
                return len(value) > 0
                
            except TypeError:

                pass
        
        return True


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "InputSelector": InputSelector,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InputSelector": node_prefix() + " INPUT SELECTOR",
}