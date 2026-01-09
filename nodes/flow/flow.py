from server import PromptServer
import comfy.model_management as mm
import time
import json
from aiohttp import web
from comfy_api.latest import io
from ...core.node import node_path, node_prefix, main_cetegory

class FlowInputSelector(io.ComfyNode):
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaInputSelector",
            display_name=node_prefix() + " Artha Input Selector",
            category=main_cetegory() + "/Flow",
            inputs=[
                io.Combo.Input(
                    "operation",
                    options=[
                        "SELECT A AS OUTPUT", 
                        "SELECT B AS OUTPUT", 
                        "NOT NULL AS OUTPUT", 
                        "BOTH TRUE A AS OUTPUT", 
                        "BOTH TRUE B AS OUTPUT",         
                    ],
                    default="SELECT A",
                ),
                io.AnyType.Input("in_a", {"forceInput": True}),
                io.AnyType.Input("in_b", {"forceInput": True}),
            ],
            outputs=[
                io.AnyType.Output(
                    "out_value",
                    display_name="out",
                    tooltip="Selected slot as output",
                ),
            ],
        )
    
    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:

        return True

    @classmethod
    def execute(cls, operation, in_a=None, in_b=None):

        if operation == "SELECT A AS OUTPUT":
            
            return (in_a,)
            
        elif operation == "SELECT B AS OUTPUT":
            
            return (in_b,)
            
        elif operation == "NOT NULL AS OUTPUT":
 
            if cls.checker(in_a) and not cls.checker(in_b):
                
                return (in_a,)
                
            elif cls.checker(in_b) and not cls.checker(in_a):
                
                return (in_b,)
                
            if cls.checker(in_a) and cls.checker(in_b):
                
                return (in_a,)
            
            else:
                
                raise ValueError("A and B not valid.")
                
        elif operation == "BOTH TRUE A AS OUTPUT":
            
            if cls.checker(in_a) and cls.checker(in_b):
                
                return (in_a,)
                
        elif operation == "BOTH TRUE B AS OUTPUT":
            
            if cls.checker(in_a) and cls.checker(in_b):
                
                return (in_b,)
                
        else:
                return (in_a,)
   
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
        
arthaPauseState = False
arthaTerminateState = False

#################################################
        
class FlowPause(io.ComfyNode):
    
    @PromptServer.instance.routes.post("/artha/flow_toggle_pause_button")
    async def toggle_pause_state(request):
        
        global arthaPauseState
               
        try:
            
            data = await request.json()
            
            # Toggle the pause state
            arthaPauseState = not arthaPauseState
            
            response_data = {
                "success": True,
                "paused": arthaPauseState,
                "message": f"Pause state changed to: {arthaPauseState}"
            }
                       
            return web.json_response(response_data)
        
        except Exception as e:
            
            return web.json_response({"success": False, "error": str(e)}, status=500) 

    def terminate():
        
        global arthaTerminateState

        if arthaTerminateState:
            
            raise mm.InterruptProcessingException(
                "Artha Flow Pause — execution terminated"
            )
  
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaFlowPause",
            display_name=node_prefix() + " Artha Flow Pause",
            category=main_cetegory() + "/Flow",
            inputs=[
                io.AnyType.Input(
                    "in_slot", 
                    display_name="in",
                    tooltip="Pass thru in slot",
                    ),
            ],
            outputs=[
                io.AnyType.Output(
                    "out_slot",
                    display_name="out",
                    tooltip="Pass thru out slot",
                ),
            ],
        )
        
    @classmethod
    def validate_inputs(cls, **kwargs) -> bool:

        return True
    
    @classmethod
    def fingerprint_inputs(cls, **kwargs):        
        
        return float("NaN")

    @classmethod
    def execute(cls, in_slot):
        
        global arthaPauseState
                              
        if arthaPauseState:
                  
            while arthaPauseState:
                
                time.sleep(0.1)
                
                if mm.processing_interrupted():
                    
                    print("Artha Flow Pause Node: ❌ Processing interrupted - Cancelling")     
                                       
                    try:
    
                        PromptServer.instance.send_sync("artha_flow_pause_button_reset", {
                            "node_class": "FlowPause",
                            "paused": arthaPauseState,
                            "message": "Reset due to interruption"
                        })
                        
                    except Exception as e:
                        
                        print(f"Artha Flow Pause Node could not send reset message to frontend: {e}")        
    
                    break
                    
                pass
    
        return (in_slot,)