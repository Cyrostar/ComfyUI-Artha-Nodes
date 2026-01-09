from datetime import datetime
import folder_paths
from comfy_api.latest import io
from ...core.node import node_path, node_prefix, main_cetegory
from ...core.helpers import filename_format

ARTHA_PROJECT_SHARED_DICT = {}

class ProjectSetupNode(io.ComfyNode):
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaProjectSetup",
            display_name=node_prefix() + " Artha Project Setup",
            category=main_cetegory() + "/Project",
            inputs=[
                io.String.Input("project_name", default="Youtube"),
                io.String.Input(
                    "project_category",
                    default="Shorts",
                    placeholder="Leave empty if not needed",
                ),
                io.String.Input(
                    "project_type",
                    default="Fun",
                    placeholder="Leave empty if not needed",
                ),
            ],
            outputs=[
                io.String.Output("project_folder"),
            ],
        )
    
    @classmethod
    def fingerprint_inputs(cls, project_name, project_category, project_type):
        
        ARTHA_PROJECT_SHARED_DICT['project_name'] = project_name
        ARTHA_PROJECT_SHARED_DICT['project_category'] = project_category
        ARTHA_PROJECT_SHARED_DICT['project_type'] = project_type
          
        checker = project_name + project_category + project_type

        return checker
    
    @classmethod
    def execute(self, project_name, project_category, project_type):
        
        project_folder = ""
        
        output_dir = folder_paths.get_output_directory()
        
        project_folder = project_folder + output_dir + "\\"
        
        if(project_name):
            
            project_folder = project_folder + filename_format(project_name) + "\\"
            
        if(project_category):
            
            project_folder = project_folder + filename_format(project_category) + "\\"

        if(project_type):
            
            project_folder = project_folder + filename_format(project_type) + "\\"
        
        return (project_folder,)
        
#################################################   
    
class ProjectPrefixNode(io.ComfyNode):
    
    CATEGORY = main_cetegory() + "/PRJ"
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaProjectPrefix",
            display_name=node_prefix() + " Artha Project Prefix",
            category=main_cetegory() + "/Project",
            inputs=[
                io.String.Input(
                    "file_title",
                    default="Artha",
                    placeholder="Leave empty if not needed",
                ),
                io.String.Input(
                    "file_model",
                    default="Flux",
                    placeholder="Leave empty if not needed",
                ),
                io.Combo.Input(
                    "file_media",
                    options=["image", "video", "none"],
                    default="image",
                ),
                io.Int.Input(
                    "file_seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                ),
            ],
            outputs=[
                io.String.Output("prefix"),
                io.Int.Output("seed"),
                io.String.Output("datetime"),
            ],
        )
    
    @classmethod
    def fingerprint_inputs(cls, file_title, file_model, file_media, file_seed):

        checker = file_title + file_model + file_media + str(file_seed)

        return checker
    
    @classmethod
    def execute(self, file_title, file_model, file_media, file_seed):
        
        prefix = ""
              
        project_name = ARTHA_PROJECT_SHARED_DICT.get('project_name')
        project_category = ARTHA_PROJECT_SHARED_DICT.get('project_category')
        project_type = ARTHA_PROJECT_SHARED_DICT.get('project_type')
              
        now = datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        if(project_name):
            prefix = prefix + filename_format(project_name) + "/"
            
        if(project_category):
            prefix = prefix + filename_format(project_category) + "/"
            
        if(project_type):
            prefix = prefix + filename_format(project_type) + "/"
            
        if(file_title):
            prefix = prefix + filename_format(file_title) + "-"
            
        prefix = prefix + "DATE-" + str(now) + "-SEED-" + str(file_seed)
        
        if(file_model):
            prefix = prefix + "-MODEL-" + filename_format(file_model)
            
        if(file_media == "image"):
            prefix = prefix + "-IMG"
            
        if(file_media == "video"):
            prefix = prefix + "-VID"
            
        if(file_media == "none"):
            prefix = prefix + "-"
            
        return (prefix, file_seed, now)
        
#################################################
        
class ProjectSeedNode(io.ComfyNode):
      
    seed_dict = {}

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaProjectSeed",
            display_name=node_prefix() +" Artha Project Seed",
            category=main_cetegory() + "/Project",
            inputs=[
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                ),
            ],
            outputs=[
                io.Int.Output(
                    "out_seed_value",
                    display_name="seed",
                    tooltip="seed"
                ),
                io.String.Output(
                    "out_seed_history",
                    display_name="history",
                    tooltip="history"
                ),
            ],
        )
                 
    @classmethod
    def fingerprint_inputs(cls, seed):
        
        checker = seed

        return checker 
    
    @classmethod    
    def execute(cls, seed):
        
        now = datetime.now()
        now = now.strftime("%H:%M:%S")
        
        cls.seed_dict[now] = seed
        
        history = ""
        
        for key, value in cls.seed_dict.items():
            
            history += "TIME " + str(key) + " SEED " + str(value) + "\n"
 
        return (seed,history)