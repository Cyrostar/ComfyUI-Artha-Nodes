from comfy_api.latest import ComfyExtension

from .nodes import NODES as ARTHA_NODES

class ComfyUIArthaNodesExtension(ComfyExtension):
      
    async def get_node_list(self):
        return [
            *ARTHA_NODES,
        ]

async def comfy_entrypoint():
    
    return ComfyUIArthaNodesExtension()
    

WEB_DIRECTORY = "web/js"