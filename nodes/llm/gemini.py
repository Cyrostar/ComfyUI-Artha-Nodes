import os
import json
import random

from PIL import Image
import numpy as np

from comfy_api.latest import io
import folder_paths

from ...core.node import node_path, node_prefix, main_cetegory
from ...core.api import load_api_key
from ...core.llm import call_gemini_text_api
from ...core.llm import call_gemini_image_api
from ...core.llm import call_gemini_tts_api
from ...core.llm import gemini_api_parameters
from ...core.llm import load_agent
from ...core.img import tensor_to_pil_image, resize_image_shortest

################################################# 

class GeminiQuestion(io.ComfyNode):
    
    DESCRIPTION = "Gemini Question node can be used to ask questions "
    DESCRIPTION += "though the context will not be preserved as "
    DESCRIPTION += "this node is not suitable for dialogue purposes."

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiQuestion",
            display_name=node_prefix() + " Artha Gemini Question",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "question",
                    display_name="question",
                    multiline=True,
                    default="",
                ),
                *gemini_api_parameters()
            ],
            outputs=[
                io.String.Output(
                "out_answer",
                display_name="answer",
                ),
            ],
        )
    
    @classmethod
    def execute(self, question, api_key, model, max_tokens, temperature):
            
        response = ""
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
            
        system_instruction = "You are an intelligent ai asistant."
            
        text_prompt = question
        
        try:
            
            # Call Gemini API
            kwargs = {
                'text_prompt'       : text_prompt,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
################################################# 

class GeminiOperation(io.ComfyNode):

    DESCRIPTION = "Gemini Operation node can be used to make "
    DESCRIPTION += "changes on the source text like replacing, "
    DESCRIPTION += "removing parts and other text operations."

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiOperation",
            display_name=node_prefix() + " Artha Gemini Operation",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "source",
                    display_name="source",
                    multiline=True,
                    default="A cat with a hat",
                ),
                io.String.Input(
                    "instruction",
                    display_name="instruction",
                    multiline=True,
                    default="Change cat to dog",
                ),
                *gemini_api_parameters(),
            ],
            outputs=[
                io.String.Output(
                    "out_response",
                    display_name="response",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        source,
        instruction,
        api_key,
        model,
        max_tokens,
        temperature
    ):

        response = ""

        # Validate API key
        if not api_key:
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            error_msg = (
                "No API key provided. Please provide an API key "
                "or set GEMINI_API_KEY environment variable."
            )
            print(error_msg)
            return (response,)

        system_instruction = (
            "Role: You are an intelligent AI assistant.\n\n"
            "Task: You will perform the given action on the text prompt.\n\n"
            f"Action: {instruction}"
        )

        text_prompt = source

        try:
            # Call Gemini API
            kwargs = {
                "text_prompt": text_prompt,
                "system_instruction": system_instruction,
                "api_key": api_key,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            response = call_gemini_text_api(**kwargs)
            return (response,)

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
################################################# 

class GeminiTranslate(io.ComfyNode):

    DESCRIPTION = "Gemini Translate node can be used to translate "
    DESCRIPTION += "one language to another."

    LANGUAGES = [
        "Chinese",
        "Spanish",
        "English",
        "Hindi",
        "Portuguese",
        "Bengali",
        "Russian",
        "Japanese",
        "Turkish",
        "German",
        "French",
        "Italian",
    ]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiTranslate",
            display_name=node_prefix() + " Artha Gemini Translate",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "text",
                    display_name="text",
                    multiline=True,
                    default="",
                ),
                io.Combo.Input(
                    "lang_from",
                    options=cls.LANGUAGES,
                    default="Chinese",
                    display_name="from",
                ),
                io.Combo.Input(
                    "lang_to",
                    options=cls.LANGUAGES,
                    default="English",
                    display_name="to",
                ),
                *gemini_api_parameters(),
            ],
            outputs=[
                io.String.Output(
                    "out_response",
                    display_name="response",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text,
        lang_from,
        lang_to,
        api_key,
        model,
        max_tokens,
        temperature
    ):

        response = ""

        # Validate API key
        if not api_key:
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            error_msg = (
                "No API key provided. Please provide an API key "
                "or set GEMINI_API_KEY environment variable."
            )
            print(error_msg)
            return (response,)

        system_instruction = (
            f"Translate the given text from {lang_from} to {lang_to}. "
            "Begin your output directly without any introductory sentence "
            "or summary phrase."
        )

        try:
            response = call_gemini_text_api(
                text_prompt=text,
                system_instruction=system_instruction,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (response,)

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return (response,)
            
#################################################

class GeminiImagen(io.ComfyNode):

    DESCRIPTION = "Gemini Imagen node is for image generation and modification."

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiImagen",
            display_name=node_prefix() + " Artha Gemini Imagen",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "text_prompt",
                    display_name="text prompt",
                    multiline=True,
                    default="A cat with a hat",
                ),
                io.Boolean.Input(
                    "modify_image",
                    display_name="modify image",
                    default=False,
                ),
                io.Image.Input(
                    "image",
                    display_name="image",
                    optional=True,
                ),
                io.String.Input(
                    "system_instruction",
                    display_name="system instruction",
                    multiline=True,
                    default="",
                    optional=True,
                ),
                *gemini_api_parameters(model="image"),
            ],
            outputs=[
                io.Image.Output(
                    "out_image",
                    display_name="image",
                ),
                io.String.Output(
                    "out_response",
                    display_name="response",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text_prompt,
        modify_image,
        system_instruction,
        api_key,
        model,
        max_tokens,
        temperature,
        image=None,
    ):

        tensor = None
        response = ""

        # Validate API key
        if not api_key:
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            print(
                "No API key provided. Please provide an API key "
                "or set GEMINI_API_KEY environment variable."
            )
            return (tensor, response)

        pil_image = None
        
        if modify_image:
            
            pil_image = image.convert("RGB")
            
        try:
                              
            # Call Gemini API
            tensor, response = call_gemini_image_api(
                text_prompt,
                pil_image,  
                system_instruction,
                api_key, 
                model, 
                max_tokens, 
                temperature
            )
        
            return (tensor, response)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (tensor, response,)
                             
#################################################

class GeminiSpeech(io.ComfyNode):

    DESCRIPTION = "Generates speech from text using the Gemini TTS model."

    # Load voice metadata once
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "speech.json")

    with open(json_path, "r", encoding="utf-8") as file:
        speech = json.load(file)

    SPEAKERS = list(speech.get("Speakers", {}).keys())
    LANGUAGES = speech.get("Languages", {})

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiSpeech",
            display_name=node_prefix() + " Artha Gemini Speech",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "text_prompt",
                    display_name="text",
                    multiline=True,
                    default="A cat with a hat",
                ),
                io.Combo.Input(
                    "voice",
                    options=cls.SPEAKERS,
                    default="Kore" if "Kore" in cls.SPEAKERS else (cls.SPEAKERS[0] if cls.SPEAKERS else ""),
                    display_name="voice",
                ),
                *gemini_api_parameters(model="tts"),
            ],
            outputs=[
                io.Audio.Output(
                    "out_audio",
                    display_name="audio",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text_prompt,
        voice,
        api_key,
        model,
        max_tokens,
        temperature,
    ):

        audio = None

        # Validate API key
        if not api_key:
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            print(
                "No API key provided. Please provide an API key "
                "or set GEMINI_API_KEY environment variable."
            )
            return (audio,)

        try:
            audio = call_gemini_tts_api(
                text_prompt=text_prompt,
                voice=voice,
                api_key=api_key,
                model=model,
                temperature=temperature,
            )

            return (audio,)

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return (audio,)

                      
#################################################

class GeminiVision(io.ComfyNode):

    DESCRIPTION = (
        "Gemini Vision node outputs a rich description "
        "of the input image."
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiVision",
            display_name=node_prefix() + " Artha Gemini Vision",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Image.Input(
                    "image",
                    display_name="image",
                ),
                io.String.Input(
                    "text_prompt",
                    display_name="prompt",
                    multiline=True,
                    default="Describe this image in detail.",
                ),
                io.String.Input(
                    "system_instruction",
                    display_name="system instruction",
                    multiline=True,
                    default="",
                    optional=True,
                ),
                *gemini_api_parameters(),
            ],
            outputs=[
                io.String.Output(
                    "out_response",
                    display_name="response",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        text_prompt,
        system_instruction,
        api_key,
        model,
        max_tokens,
        temperature,
    ):

        response = ""

        # Validate API key
        if not api_key:
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            print(
                "No API key provided. Please provide an API key "
                "or set GEMINI_API_KEY environment variable."
            )
            return (response,)

        # Load default vision system prompt if none provided
        if not system_instruction:
            
            system_instruction = load_agent("vision")

        try:
            # Convert Comfy image tensor → PIL
            pil_image = tensor_to_pil_image(image)

            response = call_gemini_text_api(
                text_prompt=text_prompt,
                image=pil_image,
                system_instruction=system_instruction,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Clean markdown artifacts if needed
            if isinstance(response, str):
                response = response.translate(
                    str.maketrans("", "", "*#")
                )

            return (response,)

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return (response,)

            
#################################################

class GeminiMotion(io.ComfyNode):

    DESCRIPTION = (
        "Gemini Motion node outputs a rich description of the input video. "
        "If the video is too large, the built-in resizer can be used to "
        "lower the size to a reasonable level."
    )

    RESIZE_OPTIONS = ["None", "480p", "360p", "240p"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiMotion",
            display_name=node_prefix() + " Artha Gemini Motion",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Image.Input(
                    "images",
                    display_name="frames",
                ),
                io.String.Input(
                    "text_prompt",
                    display_name="prompt",
                    multiline=True,
                    default="Describe this video in detail.",
                ),
                io.Combo.Input(
                    "resize",
                    options=cls.RESIZE_OPTIONS,
                    default="None",
                    display_name="resize",
                ),
                io.String.Input(
                    "system_instruction",
                    display_name="system instruction",
                    multiline=True,
                    default="",
                    optional=True,
                ),
                *gemini_api_parameters(),
            ],
            outputs=[
                io.String.Output(
                    "out_response",
                    display_name="response",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        text_prompt,
        resize,
        system_instruction,
        api_key,
        model,
        max_tokens,
        temperature,
    ):
            
        response = None
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        if not system_instruction:
            
            system_instruction = load_agent("motion")
            
        try:
            
            pil_images = []
            
            for img in images:
                
                # Convert tensor directly to base64
                imp = tensor_to_pil_image(img)
                
                if resize == "480p":
                    
                    imp = resize_image_shortest(imp, 480)
                
                if resize == "360p":
                    
                    imp = resize_image_shortest(imp, 360)
                    
                if resize == "240p":
                    
                    imp = resize_image_shortest(imp, 240)
                
                pil_images.append(imp)                
           
            # Call Gemini API
            kwargs = {
                'text_prompt'       : text_prompt,
                'image'             : pil_images,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            response = response.translate(str.maketrans("", "", "*#"))
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)

#################################################

class GeminiPrompter(io.ComfyNode):

    DESCRIPTION = "Gemini Prompter node enriches the content of your prompt."

    MEDIA_TYPES = ["IMAGE", "VIDEO"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiPrompter",
            display_name=node_prefix() + " Artha Gemini Prompter",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "text_prompt",
                    display_name="text prompt",
                    multiline=True,
                    default="A cat with a hat.",
                ),
                io.Combo.Input(
                    "media",
                    options=cls.MEDIA_TYPES,
                    default="IMAGE",
                    display_name="media",
                ),
                io.String.Input(
                    "system_instruction",
                    display_name="system instruction",
                    multiline=True,
                    default="",
                    optional=True,
                ),
                *gemini_api_parameters(),
            ],
            outputs=[
                io.String.Output(
                    "out_response",
                    display_name="response",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text_prompt,
        media,
        system_instruction,
        api_key,
        model,
        max_tokens,
        temperature,
    ):

        response = ""
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        if not system_instruction:
            
            if media == "IMAGE":
            
                system_instruction = load_agent("enrich_image")
                
            if media == "VIDEO":
            
                system_instruction = load_agent("enrich_video")
        
        try:
            
            # Call Gemini API
            kwargs = {
                'text_prompt'       : text_prompt,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
#################################################

class GeminiCondense(io.ComfyNode):

    DESCRIPTION = (
        "Gemini Question node condenses a prompt to a target word count "
        "preserving the concept."
    )

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiCondense",
            display_name=node_prefix() + " Artha Gemini Condense",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "text_prompt",
                    display_name="text prompt",
                    multiline=True,
                    default="",
                ),
                io.Int.Input(
                    "max_words",
                    display_name="max words",
                    default=400,
                    min=1,
                    max=10000,
                    step=10,
                    tooltip=(
                        "For Gemini models, a token is equivalent to about "
                        "4 characters. 100 tokens is about 60–80 English words."
                    ),
                ),
                *gemini_api_parameters(),
            ],
            outputs=[
                io.String.Output(
                    "out_response",
                    display_name="response",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text_prompt,
        max_words,
        api_key,
        model,
        max_tokens,
        temperature,
    ):
        
        response = ""
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
                
        system_instruction = "Role: You are a master of conciseness, understanding that every word carries weight and "
        system_instruction += "that unnecessary complexity can confuse AI image generators. Your expertise includes a "
        system_instruction += "deep knowledge of artistic styles, lighting techniques, and compositional theories, "
        system_instruction += "allowing you to select the most impactful descriptive language. "
        system_instruction += "You do not just shorten text; you strategically re-engineer it "
        system_instruction += "for maximum clarity and artistic output. "
        system_instruction += "Begin your output directly without any introductory sentence or summary phrase. \n\n"
        
        system_instruction += "Task: Your task is to take a user-submitted image generation prompt of any length "
        system_instruction += "and condense it into a clear, structured, and highly effective prompt of no "
        system_instruction += "more than " + str(max_words) + " words. "
        system_instruction += "The final output must be optimized for current text-to-image AI models. \n\n"
        
        try:
            
            # Call Gemini API
            kwargs = {
                'text_prompt'       : text_prompt,
                'system_instruction': system_instruction,
                'api_key'           : api_key,
                'model'             : model,
                'max_tokens'        : max_tokens,
                'temperature'       : temperature
            }
            
            response = call_gemini_text_api(**kwargs)
            
            return (response,)
            
        except Exception as e:
            
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return (response,)
            
#################################################
####################PROFILE######################
#################################################

class GeminiPortrait(io.ComfyNode):

    DESCRIPTION = (
        "Describes a character. "
        "When no image connected, output is crafted by the parameters. "
        "If image input is active, prompt is crafted from the image details. "
        "If image input is active and reconstruct is true, prompt is crafted "
        "from the image details but parameters override fetched attributes."
    )
    
    ARTHA_FACE   = io.Custom("ARTHAFACE")
    ARTHA_BODY   = io.Custom("ARTHABODY")
    ARTHA_FORM   = io.Custom("ARTHAFORM")
    ARTHA_CLOTH  = io.Custom("ARTHACLOTH")
    ARTHA_MAKEUP = io.Custom("ARTHAMAKEUP")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiPortrait",
            display_name=node_prefix() + " Artha Gemini Portrait",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "text_prompt",
                    display_name="text prompt",
                    multiline=True,
                    default="Construct a prompt describing a character in detail.",
                ),

                io.Combo.Input(
                    "identity",
                    options=["FEMININE", "MASCULINE"],
                    default="FEMININE",
                ),

                io.Combo.Input(
                    "framing",
                    options=[
                        "headshot",
                        "portrait",
                        "medium shot",
                        "wide shot",
                        "full body shot",
                    ],
                    default="portrait",
                ),

                *gemini_api_parameters(),

                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),

                io.Boolean.Input(
                    "use_image",
                    default=False,
                ),

                io.Boolean.Input(
                    "reconstruct",
                    default=False,
                ),

                io.Image.Input(
                    "image",
                    optional=True,
                ),
                cls.ARTHA_FACE.Input("face", optional=True),
                cls.ARTHA_BODY.Input("body", optional=True),
                cls.ARTHA_FORM.Input("form", optional=True),
                cls.ARTHA_CLOTH.Input("cloth", optional=True),
                cls.ARTHA_MAKEUP.Input("makeup", optional=True),
            ],
            outputs=[
                io.String.Output(
                    "out_response",
                    display_name="response",
                ),
                io.String.Output(
                    "out_traits",
                    display_name="traits",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text_prompt,
        identity,
        framing,
        api_key,
        model,
        max_tokens,
        temperature,
        seed,
        use_image,
        reconstruct,
        image=None,
        **kwargs,
    ):
        
        response = ""
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
                       
                        
        face    = kwargs.get('face',    None)
        body    = kwargs.get('body',    None)
        form    = kwargs.get('form',    None)
        makeup  = kwargs.get('makeup',  None)
        cloth   = kwargs.get('cloth',   None)
               
               
        system_instruction = "You are a professional expert in facial aesthetics, makeup, hairstyling, structural anatomy and fashion. \n\n" 
        
        if(identity == "FEMININE"):
            
            system_instruction += "Your task is to generate a clear and visually rich description of a female, "
            
        else:
            
            system_instruction += "Your task is to generate a clear and visually rich description of a male, "
        
        system_instruction += "based on a given set of attributes. \n\n"
        
        system_instruction += "Combine the provided attributes into a cohesive, elegant, and realistic description. "
        system_instruction += "Keep the tone simple, precise, and meaningful. "
        
        if(identity == "FEMININE"):
            
            system_instruction += "Use the pronoun She or Her for the subject. "
            
        else:
            
            system_instruction += "Use the pronoun He or His for the subject. "
            
        system_instruction += "Ensure the description is specific, vivid, and unambiguous. " 
        system_instruction += "Avoid redundant phrases and excessive adjectives. "
        system_instruction += "Optimize the prompt for a " + framing + " framed photo. "
        system_instruction += "Begin your output directly without any introductory sentence or summary phrase. \n\n"
                
        property_list = ""
        
        if face:
            
            if isinstance(face, dict):
            
                for key, value in face.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    if identity == "FEMININE" and key == 'HAIR_STYLE_MAS':
                        
                        continue
                        
                    if identity == "MASCULINE" and key == 'HAIR_STYLE_FEM':
                        
                        continue
                        
                    if key == "HAIR_STYLE_FEM": key = "HAIR_STYLE"
                    if key == "HAIR_STYLE_MAS": key = "HAIR_STYLE"                  
                        
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n" 
                    
            else:
                
                property_list += face
                
        if body:
            
            if isinstance(body, dict):
            
                for key, value in body.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n"

            else:
                
                property_list += body                   
                
        if form:
            
            if isinstance(form, dict):
            
                for key, value in form.items():
                    
                    if value == 'NONE':
                        
                        continue
                        
                    property_list += "- FORM " + key.replace('_', ' ') + ": " + value.upper() + "\n" 
                        
            else:
                
                property_list += form      
    
        if cloth:
            
            property_list += "\n"
            property_list += cloth
            property_list += "\n"
        
        if makeup:
            
            property_list += "\n"
            property_list += makeup
            property_list += "\n"
 
 
        if use_image and image is not None and not reconstruct:
                        
            system_instruction += "Use the provided image as your main reference. "
            system_instruction += "Carefully analyze it and describe the features listed below in detail.\n\n"
            
            system_instruction += "Features to extract: \n"
            system_instruction += "- Facial features \n"
            system_instruction += "- Body characteristics \n"
            system_instruction += "- Fitness indicators \n"
            system_instruction += "- Clothing style and appearance \n"
            system_instruction += "- Makeup details \n"
            
            try:
                
                pil_image = tensor_to_pil_image(image)
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature,
                        'seed'              : seed
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response, property_list)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response, property_list)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response, property_list)
                           
        elif use_image and image is not None and reconstruct:
            
            system_instruction += "Use the provided image as your main reference. "
            system_instruction += "Carefully analyze it and describe the features listed below in detail.\n\n"
            
            system_instruction += "Features to extract: \n"
            system_instruction += "- Facial features \n"
            system_instruction += "- Body characteristics \n"
            system_instruction += "- Fitness indicators \n"
            system_instruction += "- Clothing style and appearance \n"
            system_instruction += "- Makeup details \n\n"
            
            system_instruction += "Change the attributes you identified from the image with the ones that are listed " 
            system_instruction += "in the property list if there is a confliction. For example if the eye color you "
            system_instruction += "identified from the image is green but property list says it is blue, use the blue color. "
            system_instruction += "Also add the properties from the list which is absent from the input image. \n\n"
            
            system_instruction += "Property List \n\n"
            
            system_instruction += property_list
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            try:
                
                pil_image = tensor_to_pil_image(image)
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature,
                        'seed'              : seed
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response, property_list)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response, property_list)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response, property_list)
            
        else:
            
            system_instruction += "Property List \n\n"
            
            system_instruction += property_list
            
            system_instruction += "\n"
                       
            system_instruction += "If property list is empty return an empty response. \n\n"
               
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature,
                    'seed'              : seed
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, property_list)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, property_list)
                
#################################################
    
class GeminiFace(io.ComfyNode):

    DESCRIPTION = "Describes the face structure of the subject."
    
    ARTHA_FACE = io.Custom("ARTHAFACE")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "profile.json")

    with open(json_path, "r") as file:
        character = json.load(file)

    HEAD_TYPES          = character["HEAD"]["HEAD_TYPES"]
    HAIR_COLORS         = character["HAIR"]["HAIR_COLORS"]
    HAIR_LENGTHS        = character["HAIR"]["HAIR_LENGTHS"]
    HAIR_STYLES_FEM     = character["HAIR"]["HAIR_STYLES_FEMININE"]
    HAIR_STYLES_MAS     = character["HAIR"]["HAIR_STYLES_MASCULINE"]
    FACE_APPEAL         = character["FACE"]["FACE_APPEAL"]
    FACE_AGE            = character["FACE"]["FACE_AGE"]
    FACE_SHAPES         = character["FACE"]["FACE_SHAPES"]
    FACE_EYEBROW_TYPES  = character["FACE"]["EYEBROW_TYPES"]
    FACE_EYEBROW_SHAPES = character["FACE"]["EYEBROW_SHAPES"]
    FACE_EYE_TYPES      = character["FACE"]["EYE_TYPES"]
    FACE_EYE_SIZES      = character["FACE"]["EYE_SIZES"]
    FACE_EYE_COLORS     = character["FACE"]["EYE_COLORS"]
    FACE_NOSE_TYPES     = character["FACE"]["NOSE_TYPES"]
    FACE_LIP_TYPES      = character["FACE"]["LIP_TYPES"]
    FACE_LIP_COLORS     = character["FACE"]["LIP_COLORS"]
    FACE_EAR_TYPES      = character["FACE"]["EAR_TYPES"]
    FACE_CHEEK_TYPES    = character["FACE"]["CHEEK_TYPES"]
    FACE_CHIN_TYPES     = character["FACE"]["CHIN_TYPES"]

    @classmethod
    def define_schema(cls) -> io.Schema:

        return io.Schema(
            node_id="ArthaGeminiFace",
            display_name=node_prefix() + " Artha Gemini Face",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Combo.Input("head_type",            ["NONE"] + list(cls.HEAD_TYPES.keys()),          default="NONE"),
                io.Combo.Input("hair_color",           ["NONE"] + list(cls.HAIR_COLORS.keys()),         default="NONE"),
                io.Combo.Input("hair_length",          ["NONE"] + list(cls.HAIR_LENGTHS.keys()),        default="NONE"),
                io.Combo.Input("hair_style_fem",       ["NONE"] + list(cls.HAIR_STYLES_FEM.keys()),     default="NONE"),
                io.Combo.Input("hair_style_mas",       ["NONE"] + list(cls.HAIR_STYLES_MAS.keys()),     default="NONE"),
                io.Combo.Input("face_appeal",          ["NONE"] + list(cls.FACE_APPEAL.keys()),         default="NONE"),
                io.Combo.Input("face_age",             ["NONE"] + list(cls.FACE_AGE.keys()),            default="NONE"),
                io.Combo.Input("face_shape",           ["NONE"] + list(cls.FACE_SHAPES.keys()),         default="NONE"),
                io.Combo.Input("face_eyebrow_type",    ["NONE"] + list(cls.FACE_EYEBROW_TYPES.keys()),  default="NONE"),
                io.Combo.Input("face_eyebrow_shape",   ["NONE"] + list(cls.FACE_EYEBROW_SHAPES.keys()), default="NONE"),
                io.Combo.Input("face_eye_type",        ["NONE"] + list(cls.FACE_EYE_TYPES.keys()),     default="NONE"),
                io.Combo.Input("face_eye_size",        ["NONE"] + list(cls.FACE_EYE_SIZES.keys()),     default="NONE"),
                io.Combo.Input("face_eye_color",       ["NONE"] + list(cls.FACE_EYE_COLORS.keys()),    default="NONE"),
                io.Combo.Input("face_nose_type",       ["NONE"] + list(cls.FACE_NOSE_TYPES.keys()),    default="NONE"),
                io.Combo.Input("face_lip_type",        ["NONE"] + list(cls.FACE_LIP_TYPES.keys()),     default="NONE"),
                io.Combo.Input("face_lip_color",       ["NONE"] + list(cls.FACE_LIP_COLORS.keys()),    default="NONE"),
                io.Combo.Input("face_ear_type",        ["NONE"] + list(cls.FACE_EAR_TYPES.keys()),     default="NONE"),
                io.Combo.Input("face_cheek_type",      ["NONE"] + list(cls.FACE_CHEEK_TYPES.keys()),   default="NONE"),
                io.Combo.Input("face_chin_type",       ["NONE"] + list(cls.FACE_CHIN_TYPES.keys()),    default="NONE"),

                io.Boolean.Input("randomize", default=False),
                io.Boolean.Input("use_image", default=False),
                io.Image.Input("image", optional=True),

                *gemini_api_parameters(),
            ],
            outputs=[
                cls.ARTHA_FACE.Output("face"),
            ],
        )

    @classmethod
    def execute(cls, head_type, hair_color, hair_length, hair_style_fem, hair_style_mas, face_appeal, face_age, face_shape, face_eyebrow_type, face_eyebrow_shape, face_eye_type, face_eye_size, face_eye_color, face_nose_type, face_lip_type, face_lip_color, face_ear_type, face_cheek_type, face_chin_type, randomize, use_image, api_key, model, max_tokens, temperature, image=None, **kwargs):
              
        if use_image and image is not None:
            
            response = ""
                     
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                return (response,)
            
            text_prompt = "Describe the face in detail."
            
            system_instruction = load_agent("face")
                            
            try:
                
                pil_image = tensor_to_pil_image(image)
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response,)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response,)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
                       
        else:
            
            face_dict = {}
                         
            if randomize:
            
                face_dict['HEAD_TYPE']           = random.choice([key for key in self.HEAD_TYPES.keys()          if key != 'NONE'])
                face_dict['HAIR_COLOR']          = random.choice([key for key in self.HAIR_COLORS.keys()         if key != 'NONE'])
                face_dict['HAIR_LENGTH']         = random.choice([key for key in self.HAIR_LENGTHS.keys()        if key != 'NONE'])
                face_dict['HAIR_STYLE_FEM']      = random.choice([key for key in self.HAIR_STYLES_FEM.keys()     if key != 'NONE'])
                face_dict['HAIR_STYLE_MAS']      = random.choice([key for key in self.HAIR_STYLES_MAS.keys()     if key != 'NONE'])
                face_dict['FACE_APPEAL']         = random.choice([key for key in self.FACE_APPEAL.keys()         if key != 'NONE'])
                face_dict['FACE_AGE']            = random.choice([key for key in self.FACE_AGE.keys()            if key != 'NONE'])
                face_dict['FACE_SHAPE']          = random.choice([key for key in self.FACE_SHAPES.keys()         if key != 'NONE'])
                face_dict['FACE_EYEBROW_TYPE']   = random.choice([key for key in self.FACE_EYEBROW_TYPES.keys()  if key != 'NONE'])
                face_dict['FACE_EYEBROW_SHAPE']  = random.choice([key for key in self.FACE_EYEBROW_SHAPES.keys() if key != 'NONE'])
                face_dict['FACE_EYE_TYPE']       = random.choice([key for key in self.FACE_EYE_TYPES.keys()      if key != 'NONE'])
                face_dict['FACE_EYE_SIZE']       = random.choice([key for key in self.FACE_EYE_SIZES.keys()      if key != 'NONE'])
                face_dict['FACE_EYE_COLOR']      = random.choice([key for key in self.FACE_EYE_COLORS.keys()     if key != 'NONE'])
                face_dict['FACE_NOSE_TYPE']      = random.choice([key for key in self.FACE_NOSE_TYPES.keys()     if key != 'NONE'])
                face_dict['FACE_LIP_TYPE']       = random.choice([key for key in self.FACE_LIP_TYPES.keys()      if key != 'NONE'])
                face_dict['FACE_LIP_COLOR']      = random.choice([key for key in self.FACE_LIP_COLORS.keys()     if key != 'NONE'])
                face_dict['FACE_EAR_TYPE']       = random.choice([key for key in self.FACE_EAR_TYPES.keys()      if key != 'NONE'])
                face_dict['FACE_CHEEK_TYPE']     = random.choice([key for key in self.FACE_CHEEK_TYPES.keys()    if key != 'NONE'])
                face_dict['FACE_CHIN_TYPE']      = random.choice([key for key in self.FACE_CHIN_TYPES.keys()     if key != 'NONE'])
                
            else:
                
                face_dict['HEAD_TYPE']           = head_type
                face_dict['HAIR_COLOR']          = hair_color
                face_dict['HAIR_LENGTH']         = hair_length
                face_dict['HAIR_STYLE_FEM']      = hair_style_fem
                face_dict['HAIR_STYLE_MAS']      = hair_style_mas
                face_dict['FACE_APPEAL']         = face_appeal
                face_dict['FACE_AGE']            = face_age
                face_dict['FACE_SHAPE']          = face_shape
                face_dict['FACE_EYEBROW_TYPE']   = face_eyebrow_type
                face_dict['FACE_EYEBROW_SHAPE']  = face_eyebrow_shape
                face_dict['FACE_EYE_TYPE']       = face_eye_type
                face_dict['FACE_EYE_SIZE']       = face_eye_size
                face_dict['FACE_EYE_COLOR']      = face_eye_color
                face_dict['FACE_NOSE_TYPE']      = face_nose_type
                face_dict['FACE_LIP_TYPE']       = face_lip_type
                face_dict['FACE_LIP_COLOR']      = face_lip_color
                face_dict['FACE_EAR_TYPE']       = face_ear_type
                face_dict['FACE_CHEEK_TYPE']     = face_cheek_type
                face_dict['FACE_CHIN_TYPE']      = face_chin_type 
            
            return (face_dict,) 
  
#################################################

class GeminiBody(io.ComfyNode):

    DESCRIPTION = "Describes the body structure of the subject."
    
    ARTHA_BODY = io.Custom("ARTHABODY")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "profile.json")

    with open(json_path, "r") as file:
        character = json.load(file)

    BODY_TYPES     = character["BODY"]["BODY_TYPES"]
    BODY_HEIGHT    = character["BODY"]["HEIGHT"]
    BODY_WEIGHT    = character["BODY"]["WEIGHT"]
    BODY_BUILD     = character["BODY"]["BUILD"]
    BODY_FRAME     = character["BODY"]["FRAME"]
    BODY_SHOULDER  = character["BODY"]["SHOULDER"]
    BODY_CHEST     = character["BODY"]["CHEST"]
    BODY_BREASTS   = character["BODY"]["BREASTS"]
    BODY_TORSO     = character["BODY"]["TORSO"]
    BODY_WAIST     = character["BODY"]["WAIST"]
    BODY_HIP       = character["BODY"]["HIP"]
    BODY_LEGS      = character["BODY"]["LEGS"]
    BODY_SKIN_TONE = character["BODY"]["SKIN_TONE"]
    BODY_POSTURE   = character["BODY"]["POSTURE"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiBody",
            display_name=node_prefix() + " Artha Gemini Body",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Combo.Input("body_type",       ["NONE"] + list(cls.BODY_TYPES.keys()),     default="NONE"),
                io.Combo.Input("body_height",     ["NONE"] + list(cls.BODY_HEIGHT.keys()),    default="NONE"),
                io.Combo.Input("body_weight",     ["NONE"] + list(cls.BODY_WEIGHT.keys()),    default="NONE"),
                io.Combo.Input("body_build",      ["NONE"] + list(cls.BODY_BUILD.keys()),     default="NONE"),
                io.Combo.Input("body_frame",      ["NONE"] + list(cls.BODY_FRAME.keys()),     default="NONE"),
                io.Combo.Input("body_shoulder",   ["NONE"] + list(cls.BODY_SHOULDER.keys()),  default="NONE"),
                io.Combo.Input("body_chest",      ["NONE"] + list(cls.BODY_CHEST.keys()),     default="NONE"),
                io.Combo.Input("body_breasts",    ["NONE"] + list(cls.BODY_BREASTS.keys()),   default="NONE"),
                io.Combo.Input("body_torso",      ["NONE"] + list(cls.BODY_TORSO.keys()),     default="NONE"),
                io.Combo.Input("body_waist",      ["NONE"] + list(cls.BODY_WAIST.keys()),     default="NONE"),
                io.Combo.Input("body_hip",        ["NONE"] + list(cls.BODY_HIP.keys()),       default="NONE"),
                io.Combo.Input("body_legs",       ["NONE"] + list(cls.BODY_LEGS.keys()),      default="NONE"),
                io.Combo.Input("body_skin_tone",  ["NONE"] + list(cls.BODY_SKIN_TONE.keys()), default="NONE"),
                io.Combo.Input("body_posture",    ["NONE"] + list(cls.BODY_POSTURE.keys()),   default="NONE"),

                io.Boolean.Input("randomize", default=False),
                io.Boolean.Input("use_image", default=False),
                io.Image.Input("image", optional=True),

                *gemini_api_parameters(),
            ],
            outputs=[
                cls.ARTHA_BODY.Output("body"),
            ],
        )

    @classmethod
    def execute(cls, body_type, body_height, body_weight, body_build, body_frame, body_shoulder, body_chest, body_breasts, body_torso, body_waist, body_hip, body_legs, body_skin_tone, body_posture, randomize, use_image, api_key, model, max_tokens, temperature, image=None, **kwargs):
    
        if use_image and image is not None:
            
            response = ""
                     
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                
                return (response,)
            
            text_prompt = "Describe the body in detail."
            
            system_instruction = load_agent("body")
                            
            try:
                
                pil_image = tensor_to_pil_image(image)
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response,)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response,)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
                       
        else:
            
            body_dict = {}
            
            if randomize:
                
                body_dict['BODY_TYPES']     = random.choice([key for key in self.BODY_TYPES.keys()      if key != 'NONE'])
                body_dict['BODY_HEIGHT']    = random.choice([key for key in self.BODY_HEIGHT.keys()     if key != 'NONE'])
                body_dict['BODY_WEIGHT']    = random.choice([key for key in self.BODY_WEIGHT.keys()     if key != 'NONE'])
                body_dict['BODY_BUILD']     = random.choice([key for key in self.BODY_BUILD.keys()      if key != 'NONE'])
                body_dict['BODY_FRAME']     = random.choice([key for key in self.BODY_FRAME.keys()      if key != 'NONE'])
                body_dict['BODY_SHOULDER']  = random.choice([key for key in self.BODY_SHOULDER.keys()   if key != 'NONE'])
                body_dict['BODY_CHEST']     = random.choice([key for key in self.BODY_CHEST.keys()      if key != 'NONE'])
                body_dict['BODY_BREASTS']   = random.choice([key for key in self.BODY_BREASTS.keys()    if key != 'NONE'])
                body_dict['BODY_TORSO']     = random.choice([key for key in self.BODY_TORSO.keys()      if key != 'NONE'])
                body_dict['BODY_WAIST']     = random.choice([key for key in self.BODY_WAIST.keys()      if key != 'NONE'])
                body_dict['BODY_HIP']       = random.choice([key for key in self.BODY_HIP.keys()        if key != 'NONE'])
                body_dict['BODY_LEGS']      = random.choice([key for key in self.BODY_LEGS.keys()       if key != 'NONE'])
                body_dict['BODY_SKIN_TONE'] = random.choice([key for key in self.BODY_SKIN_TONE.keys()  if key != 'NONE'])
                body_dict['BODY_POSTURE']   = random.choice([key for key in self.BODY_POSTURE.keys()    if key != 'NONE'])
                        
            else:
    
                body_dict['BODY_TYPES']     = body_type     
                body_dict['BODY_HEIGHT']    = body_height   
                body_dict['BODY_WEIGHT']    = body_weight   
                body_dict['BODY_BUILD']     = body_build    
                body_dict['BODY_FRAME']     = body_frame    
                body_dict['BODY_SHOULDER']  = body_shoulder 
                body_dict['BODY_CHEST']     = body_chest    
                body_dict['BODY_BREASTS']   = body_breasts  
                body_dict['BODY_TORSO']     = body_torso    
                body_dict['BODY_WAIST']     = body_waist    
                body_dict['BODY_HIP']       = body_hip      
                body_dict['BODY_LEGS']      = body_legs     
                body_dict['BODY_SKIN_TONE'] = body_skin_tone
                body_dict['BODY_POSTURE']   = body_posture  
                   
            return (body_dict,) 
  
#################################################

class GeminiForm(io.ComfyNode):

    DESCRIPTION = "Describes fitness of the subject."
    
    ARTHA_FORM = io.Custom("ARTHAFORM")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "profile.json")

    with open(json_path, "r") as file:
        character = json.load(file)

    FORM = character["FORM"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiForm",
            display_name=node_prefix() + " Artha Gemini Form",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Boolean.Input("chest",      default=False),
                io.Boolean.Input("shoulders",  default=False),
                io.Boolean.Input("arms",       default=False),
                io.Boolean.Input("biceps",     default=False),
                io.Boolean.Input("triceps",    default=False),
                io.Boolean.Input("forearms",   default=False),
                io.Boolean.Input("abs",        default=False),
                io.Boolean.Input("core",       default=False),
                io.Boolean.Input("obliques",   default=False),
                io.Boolean.Input("back",       default=False),
                io.Boolean.Input("lats",       default=False),
                io.Boolean.Input("traps",      default=False),
                io.Boolean.Input("legs",       default=False),
                io.Boolean.Input("quadriceps", default=False),
                io.Boolean.Input("hamstrings", default=False),
                io.Boolean.Input("calves",     default=False),
                io.Boolean.Input("glutes",     default=False),

                io.Boolean.Input("use_image", default=False),
                io.Image.Input("image", optional=True),

                *gemini_api_parameters(),
            ],
            outputs=[
                cls.ARTHA_FORM.Output("form"),
            ],
        )

    @classmethod
    def execute(cls, chest, shoulders, arms, biceps, triceps, forearms, abs, core, obliques, back, lats, traps, legs, quadriceps, hamstrings, calves, glutes, use_image, api_key, model, max_tokens, temperature, image=None, **kwargs):
        
        if use_image and image is not None:
            
            response = ""
                     
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                
                return (response,)
            
            text_prompt = "Describe the fitness in detail."
            
            system_instruction = load_agent("form")
                            
            try:
                
                pil_image = tensor_to_pil_image(image)
                
                try:
                    
                    # Call Gemini API
                    kwargs = {
                        'text_prompt'       : text_prompt,
                        'image'             : pil_image,
                        'system_instruction': system_instruction,
                        'api_key'           : api_key,
                        'model'             : model,
                        'max_tokens'        : max_tokens,
                        'temperature'       : temperature
                    }
                    
                    response = call_gemini_text_api(**kwargs)
                    
                    response = response.translate(str.maketrans("", "", "*#"))
                    
                    return (response,)
                    
                except Exception as e:
                    
                    error_msg = f"Error processing request: {str(e)}"
                    print(error_msg)
                    return (response,)
                
            except Exception as e:
                
                print(f"Error opening image: {e}")
                return (response,)
                       
        else:
            
            form_dict = {}
                
            if chest      : form_dict['CHEST']      = self.FORM['CHEST']     
            if shoulders  : form_dict['SHOULDERS']  = self.FORM['SHOULDERS'] 
            if arms       : form_dict['ARMS']       = self.FORM['ARMS']      
            if biceps     : form_dict['BICEPS']     = self.FORM['BICEPS']   
            if triceps    : form_dict['TRICEPS']    = self.FORM['TRICEPS']  
            if forearms   : form_dict['FOREARMS']   = self.FORM['FOREARMS'] 
            if abs        : form_dict['ABS']        = self.FORM['ABS']      
            if core       : form_dict['CORE']       = self.FORM['CORE']     
            if obliques   : form_dict['OBLIQUES']   = self.FORM['OBLIQUES'] 
            if back       : form_dict['BACK']       = self.FORM['BACK']      
            if lats       : form_dict['LATS']       = self.FORM['LATS']     
            if traps      : form_dict['TRAPS']      = self.FORM['TRAPS']    
            if legs       : form_dict['LEGS']       = self.FORM['LEGS']     
            if quadriceps : form_dict['QUADRICEPS'] = self.FORM['QUADRICEPS']
            if hamstrings : form_dict['HAMSTRINGS'] = self.FORM['HAMSTRINGS']
            if calves     : form_dict['CALVES']     = self.FORM['CALVES']    
            if glutes     : form_dict['GLUTES']     = self.FORM['GLUTES']  
            
            return (form_dict,)

#################################################  

class GeminiCloth(io.ComfyNode):

    DESCRIPTION = "Describes the clothing of the subject."
    
    ARTHA_CLOTH = io.Custom("ARTHACLOTH")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiCloth",
            display_name=node_prefix() + " Artha Gemini Cloth",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Image.Input("image"),
                *gemini_api_parameters(),
            ],
            outputs=[
                cls.ARTHA_CLOTH.Output("cloth"),
            ],
        )

    @classmethod
    def execute(cls, image, api_key, model, max_tokens, temperature, **kwargs):
        
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        text_prompt = "Identify the clothes and list each one."
        
        system_instruction = load_agent("cloth")
                            
        try:
            
            pil_image = tensor_to_pil_image(image)
            
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response,)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response,)
            
        except Exception as e:
            
            print(f"Error opening image: {e}")
            return (response,)
            
#################################################

class GeminiMakeup(io.ComfyNode):

    DESCRIPTION = "Describes the make-up of the subject."
    
    ARTHA_MAKEUP = io.Custom("ARTHAMAKEUP")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiMakeup",
            display_name=node_prefix() + " Artha Gemini Makeup",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Image.Input("image"),
                *gemini_api_parameters(),
            ],
            outputs=[
                cls.ARTHA_MAKEUP.Output("makeup"),
            ],
        )

    @classmethod
    def execute(cls, image, api_key, model, max_tokens, temperature, **kwargs):
        
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        text_prompt = "Identify the make-up of the face and list each element."
        
        system_instruction = load_agent("makeup")
                            
        try:
            
            pil_image = tensor_to_pil_image(image)
            
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response,)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response,)
            
        except Exception as e:
            
            print(f"Error opening image: {e}")
            return (response,)
            
#################################################

class GeminiBackdrop:
    
    CATEGORY = main_cetegory() + "/LLM/GEMINI"
    DESCRIPTION = "Describes the background of an image."
    
    @classmethod
    def INPUT_TYPES(self):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                **gemini_api_parameters(),
                "image": (sorted(files), {"image_upload": True}),                
            },
        }
           
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("backdrop",)
    FUNCTION = "artha_main"

    def artha_main(self, image, api_key, model, max_tokens, temperature, **kwargs):
        
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        text_prompt = "Describe the background."
        
        system_instruction = load_agent("backdrop")
                            
        try:
            
            pil_image = tensor_to_pil_image(image)
            
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response,)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response,)
            
        except Exception as e:
            
            print(f"Error opening image: {e}")
            return (response,)
               
#################################################
####################COMPOSE######################
#################################################

class GeminiCompose(io.ComfyNode):

    DESCRIPTION = (
        "Describes a character. "
        "When no image connected, output is crafted by the parameters. "
        "If image input is active, prompt is crafted from the image details. "
        "If image input is active and reconstruct is true, prompt is crafted "
        "from the image details but parameters override fetched attributes."
    )

    # Typed custom inputs
    ARTHA_SUBJECT = io.Custom("ARTHASUBJECT")
    ARTHA_SCENE   = io.Custom("ARTHASCENERY")
    ARTHA_CAMERA  = io.Custom("ARTHACAM")
    ARTHA_LIGHT   = io.Custom("ARTHALIGHT")
    ARTHA_STYLE   = io.Custom("ARTHASTYLE")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiCompose",
            display_name=node_prefix() + " Artha Gemini Compose",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "text_prompt",
                    multiline=True,
                    default="Construct a prompt describing an image composition in detail.",
                ),

                *gemini_api_parameters(),

                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),

                io.Boolean.Input(
                    "use_image",
                    default=False,
                ),

                io.Boolean.Input(
                    "reconstruct",
                    default=False,
                ),

                io.Image.Input(
                    "image",
                    optional=True,
                ),

                cls.ARTHA_SUBJECT.Input("subject", optional=True),
                cls.ARTHA_SCENE.Input("scene", optional=True),
                cls.ARTHA_CAMERA.Input("camera", optional=True),
                cls.ARTHA_LIGHT.Input("light", optional=True),
                cls.ARTHA_STYLE.Input("style", optional=True),
            ],
            outputs=[
                io.String.Output(
                    "out_response",
                    display_name="response",
                ),
                io.String.Output(
                    "out_traits",
                    display_name="traits",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text_prompt,
        api_key,
        model,
        max_tokens,
        temperature,
        seed,
        use_image,
        reconstruct,
        image=None,
        **kwargs,
    ):
        
        response = ""
        
        # Validate API key       
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
                       
        subject   = kwargs.get('subject',   None)
        scene     = kwargs.get('scene',     None)         
        camera    = kwargs.get('camera',    None)
        light     = kwargs.get('light',     None)
        style     = kwargs.get('style',     None)
             
        system_instruction = "You are a professional expert in visual composition. \n\n" 
                
        system_instruction += "Your task is to generate a clear and visually rich description of an "        
        system_instruction += "image composition based on a given set of attributes. \n\n"
        
        system_instruction += "Combine the provided attributes into a cohesive, elegant, and realistic description. "
        system_instruction += "Keep the tone simple, precise, and meaningful. "           
        system_instruction += "Ensure the description is specific, vivid, and unambiguous. " 
        system_instruction += "Avoid redundant phrases and excessive adjectives. "
        system_instruction += "Begin your output directly without any introductory sentence or summary phrase. \n\n"
                
        property_list = ""
        
        if subject:
            
            property_list += subject
                       
        if scene:
            
            if property_list: property_list += "\n\n"  
            
            if isinstance(scene, dict):
            
                for key, value in scene.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n"

            else:
                
                property_list += scene 
                
        if camera:
            
            if property_list: property_list += "\n\n" 
            
            if isinstance(camera, dict):
            
                for key, value in camera.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n"

            else:
                
                property_list += camera 

        if light:
            
            if property_list: property_list += "\n\n" 
            
            if isinstance(light, dict):
            
                for key, value in light.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n"

            else:
                
                property_list += light  

        if style:
            
            if property_list: property_list += "\n\n" 
            
            if isinstance(style, dict):
            
                for key, value in style.items():
                    
                    if value == 'NONE':
                        
                        continue
                    
                    property_list += "- " + key.replace('_', ' ') + ": " + value + "\n"

            else:
                
                property_list += style                  

        if use_image and image is not None and not reconstruct:
                        
            system_instruction += "Use the provided image as your main reference. "
            system_instruction += "Carefully analyze it and describe the features listed below in detail.\n\n"
            
            system_instruction += "Features to extract: \n"
            system_instruction += "- Subject(s) \n"
            system_instruction += "- Scene and background \n"
            system_instruction += "- Camera settings and framing \n"
            system_instruction += "- Light \n"
            system_instruction += "- Style \n"
            system_instruction += "- Atmosphere \n"
            system_instruction += "- Mood \n\n"
            
            system_instruction += "Include face, hair, makeup, body, fitness, pose, and clothing of the main subjects(s). " 
            system_instruction += "Also include the interactions if the subjects are more than one. \n\n" 
              
            pil_image = tensor_to_pil_image(image)
                
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature,
                    'seed'              : seed
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, property_list)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, property_list)
                
                
        elif use_image and image is not None and reconstruct:
            
            system_instruction += "Use the provided image as your main reference. "
            system_instruction += "Carefully analyze it and describe the features listed below in detail.\n\n"
            
            system_instruction += "Features to extract: \n"
            system_instruction += "- Subject(s) \n"
            system_instruction += "- Scene and background \n"
            system_instruction += "- Camera settings and framing \n"
            system_instruction += "- Light \n"
            system_instruction += "- Style \n"
            system_instruction += "- Atmosphere \n"
            system_instruction += "- Mood \n\n"
            
            system_instruction += "Include face, hair, makeup, body, fitness, pose, and clothing of the main subjects(s). " 
            system_instruction += "Also include the interactions if the subjects are more than one. \n\n" 
            
            system_instruction += "Change the attributes you identified from the image with the ones that are listed " 
            system_instruction += "in the property list if there is a confliction. For example if the eye color you "
            system_instruction += "identified from the image is green but property list says it is blue, use the blue color. "
            system_instruction += "Also add the properties from the list which is absent from the input image. \n\n"
            
            system_instruction += "Property List \n\n"
            
            system_instruction += property_list
            
            pil_image = tensor_to_pil_image(image)
            
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature,
                    'seed'              : seed
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, property_list)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, property_list)
            

        else:
            
            system_instruction += "Property List \n\n"
            
            system_instruction += property_list
            
            system_instruction += "\n"
                       
            system_instruction += "If property list is empty return an empty response. \n\n"
               
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature,
                    'seed'              : seed
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, property_list)               
            
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, property_list)
                
#################################################

class GeminiSubject(io.ComfyNode):

    DESCRIPTION = (
        "Subject settings for image or video prompting. "
        "If use image is selected the settings are fetched "
        "from the uploaded image by Gemini Vision."
    )

    ARTHA_SUBJECT = io.Custom("ARTHASUBJECT")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiSubject",
            display_name=node_prefix() + " Artha Gemini Subject",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "text_prompt",
                    multiline=True,
                    default="A cat with a hat",
                ),

                *gemini_api_parameters(),

                io.Boolean.Input(
                    "use_image",
                    default=False,
                ),

                io.Boolean.Input(
                    "only_main",
                    default=True,
                ),

                io.Image.Input(
                    "image",
                    optional=True,
                ),
            ],
            outputs=[
                cls.ARTHA_SUBJECT.Output(
                    "subject",
                ),
            ],
        )

    @classmethod
    def execute(cls, text_prompt, api_key, model, max_tokens, temperature, use_image, only_main, image=None, **kwargs):
    
        response = ""
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response,)
        
        if use_image and image is not None:
            
            text_prompt = "Describe the subject in detail."
            
            if only_main:
            
                system_instruction = load_agent("subject")
                
            else:
            
                system_instruction = load_agent("subjects")
             
            pil_image = tensor_to_pil_image(image)
                
            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response,)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response,)
                                    
        else:
            
            response = text_prompt
            
            return (response,)
            
################################################# 

class GeminiScenery(io.ComfyNode):

    DESCRIPTION = (
        "Scene settings for image or video prompting. "
        "If use image is selected the settings are fetched "
        "from the uploaded image by Gemini Vision."
    )

    ARTHA_SCENERY = io.Custom("ARTHASCENERY")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "compose.json")

    with open(json_path, "r") as file:
        scenery = json.load(file)

    markdown = "# SCENE OPTIONS\n\n"

    for k, v in scenery["SCENERY"].items():
        markdown += f"## {k}\n"
        if isinstance(v, dict):
            for kk, vv in v.items():
                markdown += f"- {kk}: {vv}\n"
        else:
            markdown += f"### {v}\n"
        markdown += "\n"

    SCENERY_LANDSCAPES    = scenery["SCENERY"]["LANDSCAPES"]
    SCENERY_URBAN         = scenery["SCENERY"]["URBAN"]
    SCENERY_INTERIOR      = scenery["SCENERY"]["INTERIOR"]
    SCENERY_FANTASY       = scenery["SCENERY"]["FANTASY"]
    SCENERY_FUTURISTIC    = scenery["SCENERY"]["FUTURISTIC"]
    SCENERY_ABSTRACT      = scenery["SCENERY"]["ABSTRACT"]
    SCENERY_MISC          = scenery["SCENERY"]["MISCELLANEOUS"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiScenery",
            display_name=node_prefix() + " Artha Gemini Scenery",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Combo.Input(
                    "landscapes",
                    options=["NONE"] + list(cls.SCENERY_LANDSCAPES.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "urban",
                    options=["NONE"] + list(cls.SCENERY_URBAN.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "interior",
                    options=["NONE"] + list(cls.SCENERY_INTERIOR.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "fantasy",
                    options=["NONE"] + list(cls.SCENERY_FANTASY.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "futuristic",
                    options=["NONE"] + list(cls.SCENERY_FUTURISTIC.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "abstract",
                    options=["NONE"] + list(cls.SCENERY_ABSTRACT.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "miscellaneous",
                    options=["NONE"] + list(cls.SCENERY_MISC.keys()),
                    default="NONE",
                ),

                *gemini_api_parameters(),

                io.Boolean.Input(
                    "use_image",
                    default=False,
                ),

                io.Image.Input(
                    "image",
                    optional=True,
                ),
            ],
            outputs=[
                cls.ARTHA_SCENERY.Output(
                    "scene",
                ),
                io.String.Output(
                    "markdown",
                ),
            ],
        )

    @classmethod
    def execute(cls, landscapes, urban, interior, fantasy, futuristic, abstract, miscellaneous, api_key, model, max_tokens, temperature, use_image, image, **kwargs):
    
        response = ""
        
        markdown = cls.markdown
        
        # Validate API key
        if not api_key:
            
            api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            
            error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
            print(error_msg)
            
            return (response, markdown)
        
        if use_image and image is not None:
            
            text_prompt = "Describe the scene in detail."
            
            system_instruction = load_agent("scenery")
             
            pil_image = tensor_to_pil_image(image)
                
            try:           
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                    
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, markdown)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, markdown)
                   
        else:
            
            text_prompt = ""
            
            if landscapes       != "NONE": text_prompt += cls.SCENERY_LANDSCAPES[landscapes]
            if urban            != "NONE": text_prompt += cls.SCENERY_URBAN[urban]
            if interior         != "NONE": text_prompt += cls.SCENERY_INTERIOR[interior]
            if fantasy          != "NONE": text_prompt += cls.SCENERY_FANTASY[fantasy]
            if futuristic       != "NONE": text_prompt += cls.SCENERY_FUTURISTIC[futuristic]
            if abstract         != "NONE": text_prompt += cls.SCENERY_ABSTRACT[abstract]
            if miscellaneous    != "NONE": text_prompt += cls.SCENERY_MISCELLANEOUS[miscellaneous]
            
            system_instruction = load_agent("scene")

            try:
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                    
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, markdown)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, markdown)
            
#################################################

class GeminiCamera(io.ComfyNode):

    DESCRIPTION = (
        "Camera settings for image or video prompting. "
        "If use image is selected the settings are fetched "
        "from the uploaded image by Gemini Vision."
    )

    ARTHA_CAMERA = io.Custom("ARTHACAM")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "compose.json")

    with open(json_path, "r") as file:
        camera = json.load(file)

    markdown = "# CAMERA OPTIONS\n\n"

    for k, v in camera["CAMERA"].items():
        markdown += f"## {k}\n"
        if isinstance(v, dict):
            for kk, vv in v.items():
                markdown += f"- {kk}: {vv}\n"
        else:
            markdown += f"### {v}\n"
        markdown += "\n"

    TYPE           = camera["CAMERA"]["TYPE"]
    CONTEXT        = camera["CAMERA"]["CONTEXT"]
    SHOT           = camera["CAMERA"]["SHOT"]
    LENSES         = camera["CAMERA"]["LENSES"]
    ANGLES         = camera["CAMERA"]["ANGLES"]
    MOTION         = camera["CAMERA"]["MOTION"]
    FOCAL_LENGTH   = camera["CAMERA"]["FOCAL_LENGTH"]
    DEPTH_OF_FIELD = camera["CAMERA"]["DEPTH_OF_FIELD"]
    ASPECT_RATIO   = camera["CAMERA"]["ASPECT_RATIO"]
    SENSOR_TYPE    = camera["CAMERA"]["SENSOR_TYPE"]
    CAMERA_STYLE   = camera["CAMERA"]["CAMERA_STYLE"]
    FRAMING        = camera["CAMERA"]["FRAMING"]
    CAMERA_MODELS  = camera["CAMERA"]["CAMERA_MODELS"]
    FILMSTOCK      = camera["CAMERA"]["FILMSTOCK"]
    COLOR          = camera["CAMERA"]["COLOR"]
    TEXTURE        = camera["CAMERA"]["TEXTURE"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiCamera",
            display_name=node_prefix() + " Artha Gemini Camera",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Combo.Input("type",           ["NONE"] + list(cls.TYPE.keys()),           default="NONE"),
                io.Combo.Input("context",        ["NONE"] + list(cls.CONTEXT.keys()),        default="NONE"),
                io.Combo.Input("shot",           ["NONE"] + list(cls.SHOT.keys()),           default="NONE"),
                io.Combo.Input("lenses",         ["NONE"] + list(cls.LENSES.keys()),         default="NONE"),
                io.Combo.Input("angles",         ["NONE"] + list(cls.ANGLES.keys()),         default="NONE"),
                io.Combo.Input("motion",         ["NONE"] + list(cls.MOTION.keys()),         default="NONE"),
                io.Combo.Input("focal_length",   ["NONE"] + list(cls.FOCAL_LENGTH.keys()),   default="NONE"),
                io.Combo.Input("depth_of_field", ["NONE"] + list(cls.DEPTH_OF_FIELD.keys()), default="NONE"),
                io.Combo.Input("aspect_ratio",   ["NONE"] + list(cls.ASPECT_RATIO.keys()),   default="NONE"),
                io.Combo.Input("sensor_type",    ["NONE"] + list(cls.SENSOR_TYPE.keys()),    default="NONE"),
                io.Combo.Input("camera_style",   ["NONE"] + list(cls.CAMERA_STYLE.keys()),   default="NONE"),
                io.Combo.Input("framing",        ["NONE"] + list(cls.FRAMING.keys()),        default="NONE"),
                io.Combo.Input("camera_models",  ["NONE"] + list(cls.CAMERA_MODELS.keys()),  default="NONE"),
                io.Combo.Input("filmstock",      ["NONE"] + list(cls.FILMSTOCK.keys()),      default="NONE"),
                io.Combo.Input("color",          ["NONE"] + list(cls.COLOR.keys()),          default="NONE"),
                io.Combo.Input("texture",        ["NONE"] + list(cls.TEXTURE.keys()),        default="NONE"),

                *gemini_api_parameters(),

                io.Boolean.Input(
                    "randomize",
                    default=False,
                ),

                io.Boolean.Input(
                    "use_image",
                    default=False,
                ),

                io.Image.Input(
                    "image",
                    optional=True,
                ),
            ],
            outputs=[
                cls.ARTHA_CAMERA.Output("camera"),
                io.String.Output("markdown"),
            ],
        )

    @classmethod
    def execute(cls, type, context, shot, lenses, angles, motion, focal_length,depth_of_field, aspect_ratio,sensor_type, camera_style, framing, camera_models, filmstock, color, texture, api_key, model, max_tokens, temperature, randomize, use_image, image=None, **kwargs):
        
        response = ""
            
        markdown = cls.markdown
       
        if use_image and image is not None:
                                
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                
                return (response, markdown)
            
            text_prompt = "Describe the camera settings in detail."
            
            system_instruction = load_agent("camera")
                     
            pil_image = tensor_to_pil_image(image)
            
            try:           
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                    
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, markdown)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, markdown)
                                        
        else:
            
            cam_dict = {}
            
            if randomize:
                cam_dict['TYPE']            = random.choice([key for key in cls.TYPE           .keys()      if key != 'NONE'])
                cam_dict['CONTEXT']         = random.choice([key for key in cls.CONTEXT        .keys()      if key != 'NONE'])
                cam_dict['SHOT']            = random.choice([key for key in cls.SHOT           .keys()      if key != 'NONE'])
                cam_dict['LENSES']          = random.choice([key for key in cls.LENSES         .keys()      if key != 'NONE'])
                cam_dict['ANGLES']          = random.choice([key for key in cls.ANGLES         .keys()      if key != 'NONE'])
                cam_dict['MOTION']          = random.choice([key for key in cls.MOTION         .keys()      if key != 'NONE'])
                cam_dict['FOCAL_LENGTH']    = random.choice([key for key in cls.FOCAL_LENGTH   .keys()      if key != 'NONE'])
                cam_dict['DEPTH_OF_FIELD']  = random.choice([key for key in cls.DEPTH_OF_FIELD .keys()      if key != 'NONE'])
                cam_dict['ASPECT_RATIO']    = random.choice([key for key in cls.ASPECT_RATIO   .keys()      if key != 'NONE'])
                cam_dict['SENSOR_TYPE']     = random.choice([key for key in cls.SENSOR_TYPE    .keys()      if key != 'NONE'])
                cam_dict['CAMERA_STYLE']    = random.choice([key for key in cls.CAMERA_STYLE   .keys()      if key != 'NONE'])
                cam_dict['FRAMING']         = random.choice([key for key in cls.FRAMING        .keys()      if key != 'NONE'])
                cam_dict['CAMERA_MODELS']   = random.choice([key for key in cls.CAMERA_MODELS  .keys()      if key != 'NONE'])
                cam_dict['FILMSTOCK']       = random.choice([key for key in cls.FILMSTOCK      .keys()      if key != 'NONE'])
                cam_dict['COLOR']           = random.choice([key for key in cls.COLOR          .keys()      if key != 'NONE'])
                cam_dict['TEXTURE']         = random.choice([key for key in cls.TEXTURE        .keys()      if key != 'NONE'])
                        
            else:
    
                cam_dict['TYPE']            = type
                cam_dict['CONTEXT']         = context
                cam_dict['SHOT']            = shot     
                cam_dict['LENSES']          = lenses          
                cam_dict['ANGLES']          = angles          
                cam_dict['MOTION']          = motion          
                cam_dict['FOCAL_LENGTH']    = focal_length    
                cam_dict['DEPTH_OF_FIELD']  = depth_of_field  
                cam_dict['ASPECT_RATIO']    = aspect_ratio    
                cam_dict['SENSOR_TYPE']     = sensor_type     
                cam_dict['CAMERA_STYLE']    = camera_style    
                cam_dict['FRAMING']         = framing         
                cam_dict['CAMERA_MODELS']   = camera_models   
                cam_dict['FILMSTOCK']       = filmstock                 
                cam_dict['COLOR']           = color           
                cam_dict['TEXTURE']         = texture                         
                   
            return (cam_dict, markdown)
               
################################################# 

class GeminiLight(io.ComfyNode):

    DESCRIPTION = (
        "Light settings for image or video prompting. "
        "If use image is selected the settings are fetched "
        "from the uploaded image by Gemini Vision."
    )

    ARTHA_LIGHT = io.Custom("ARTHALIGHT")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "compose.json")

    with open(json_path, "r") as file:
        light = json.load(file)

    markdown = "# LIGHT OPTIONS\n\n"

    for k, v in light["LIGHT"].items():
        markdown += f"## {k}\n"
        if isinstance(v, dict):
            for kk, vv in v.items():
                markdown += f"- {kk}: {vv}\n"
        else:
            markdown += f"### {v}\n"
        markdown += "\n"

    LIGHT_SOURCE    = light["LIGHT"]["SOURCE"]
    LIGHT_QUALITY   = light["LIGHT"]["QUALITY"]
    LIGHT_DIRECTION = light["LIGHT"]["DIRECTION"]
    LIGHT_TIMEOFDAY = light["LIGHT"]["TIMEOFDAY"]
    LIGHT_PHENOMENA = light["LIGHT"]["PHENOMENA"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiLight",
            display_name=node_prefix() + " Artha Gemini Light",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Combo.Input(
                    "source",
                    ["NONE"] + list(cls.LIGHT_SOURCE.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "quality",
                    ["NONE"] + list(cls.LIGHT_QUALITY.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "direction",
                    ["NONE"] + list(cls.LIGHT_DIRECTION.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "timeofday",
                    ["NONE"] + list(cls.LIGHT_TIMEOFDAY.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "phenomena",
                    ["NONE"] + list(cls.LIGHT_PHENOMENA.keys()),
                    default="NONE",
                ),

                *gemini_api_parameters(),

                io.Boolean.Input(
                    "randomize",
                    default=False,
                ),

                io.Boolean.Input(
                    "use_image",
                    default=False,
                ),

                io.Image.Input(
                    "image",
                    optional=True,
                ),
            ],
            outputs=[
                cls.ARTHA_LIGHT.Output("light"),
                io.String.Output("markdown"),
            ],
        )
    
    @classmethod
    def execute(cls, source, quality, direction, timeofday, phenomena, api_key, model, max_tokens, temperature, randomize, use_image, image=None, **kwargs):
    
        response = ""
            
        markdown = cls.markdown
        
        if use_image and image is not None:
                     
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                
                return (response, markdown)
            
            text_prompt = "Describe the light settings in detail."
            
            system_instruction = load_agent("light")
                            
            pil_image = tensor_to_pil_image(image)
            
            try:           
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                    
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, markdown)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, markdown)

        else:
            
            light_dict = {}
            
            if randomize:
                
                light_dict['LIGHT_SOURCE']      = random.choice([key for key in cls.LIGHT_SOURCE     .keys()      if key != 'NONE'])
                light_dict['LIGHT_QUALITY']     = random.choice([key for key in cls.LIGHT_QUALITY    .keys()      if key != 'NONE'])
                light_dict['LIGHT_DIRECTION']   = random.choice([key for key in cls.LIGHT_DIRECTION  .keys()      if key != 'NONE'])
                light_dict['LIGHT_TIMEOFDAY']   = random.choice([key for key in cls.LIGHT_TIMEOFDAY  .keys()      if key != 'NONE'])
                light_dict['LIGHT_PHENOMENA']   = random.choice([key for key in cls.LIGHT_PHENOMENA  .keys()      if key != 'NONE'])
                
            else:
    
                light_dict['LIGHT_SOURCE']      = source   
                light_dict['LIGHT_QUALITY']     = quality 
                light_dict['LIGHT_DIRECTION']   = direction 
                light_dict['LIGHT_TIMEOFDAY']   = timeofday
                light_dict['LIGHT_PHENOMENA']   = phenomena          
                   
            return (light_dict, markdown) 
               
################################################# 

class GeminiStyle(io.ComfyNode):

    DESCRIPTION = (
        "Style settings for image or video prompting. "
        "If use image is selected the settings are fetched "
        "from the uploaded image by Gemini Vision."
    )

    ARTHA_STYLE = io.Custom("ARTHASTYLE")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "json", "compose.json")

    with open(json_path, "r") as file:
        style = json.load(file)

    markdown = "# STYLE OPTIONS\n\n"

    for k, v in style["STYLE"].items():
        markdown += f"## {k}\n"
        if isinstance(v, dict):
            for kk, vv in v.items():
                markdown += f"- {kk}: {vv}\n"
        else:
            markdown += f"### {v}\n"
        markdown += "\n"

    STYLE_TRADITIONAL   = style["STYLE"]["TRADITIONAL"]
    STYLE_MODERN        = style["STYLE"]["MODERN"]
    STYLE_PHOTOGRAPHIC  = style["STYLE"]["PHOTOGRAPHIC"]
    STYLE_NAMED         = style["STYLE"]["NAMED"]
    STYLE_INSPIRED      = style["STYLE"]["INSPIRED"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiStyle",
            display_name=node_prefix() + " Artha Gemini Style",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.Combo.Input(
                    "traditional",
                    ["NONE"] + list(cls.STYLE_TRADITIONAL.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "modern",
                    ["NONE"] + list(cls.STYLE_MODERN.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "photographic",
                    ["NONE"] + list(cls.STYLE_PHOTOGRAPHIC.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "named",
                    ["NONE"] + list(cls.STYLE_NAMED.keys()),
                    default="NONE",
                ),
                io.Combo.Input(
                    "inspired",
                    ["NONE"] + list(cls.STYLE_INSPIRED.keys()),
                    default="NONE",
                ),

                *gemini_api_parameters(),

                io.Boolean.Input(
                    "use_image",
                    default=False,
                ),

                io.Image.Input(
                    "image",
                    optional=True,
                ),
            ],
            outputs=[
                cls.ARTHA_STYLE.Output("style"),
                io.String.Output("markdown"),
            ],
        )
        
    @classmethod
    def execute(cls, traditional, modern, photographic, named, inspired, api_key, model, max_tokens, temperature, use_image, image=None, **kwargs):
    
        response = ""
        
        markdown = cls.markdown
        
        if use_image and image is not None:
                           
            # Validate API key
            if not api_key:
                
                api_key = load_api_key("gemini") or os.environ.get("GEMINI_API_KEY")
                
            if not api_key:
                
                error_msg = "No API key provided. Please provide an API key or set GEMINI_API_KEY environment variable."
                print(error_msg)
                
                return (response, markdown)
            
            text_prompt = "Describe the style in detail."
            
            system_instruction = load_agent("style")

            pil_image = tensor_to_pil_image(image)
            
            try:           
                
                # Call Gemini API
                kwargs = {
                    'text_prompt'       : text_prompt,
                    'image'             : pil_image,
                    'system_instruction': system_instruction,
                    'api_key'           : api_key,
                    'model'             : model,
                    'max_tokens'        : max_tokens,
                    'temperature'       : temperature
                }
                
                response = call_gemini_text_api(**kwargs)
                    
                response = response.translate(str.maketrans("", "", "*#"))
                
                return (response, markdown)
                
            except Exception as e:
                
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                return (response, markdown)
                                      
        else:
            
            style_dict = {}
      
            style_dict['STYLE_TRADITIONAL ']    = traditional
            style_dict['STYLE_MODERN']          = modern     
            style_dict['STYLE_PHOTOGRAPHIC']    = photographic
            style_dict['STYLE_NAMED']           = named     
            style_dict['STYLE_INSPIRED']        = inspired          
                   
            return (style_dict, markdown)
                         
#################################################
####################DISPLAY######################
#################################################  

class GeminiResponse(io.ComfyNode):
       
    DESCRIPTION = "Gemini Response Node's objective is to display other Gemini nodes's outputs."
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiResponse",
            display_name=node_prefix() + " Artha Gemini Response",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "response",
                    display_name="response",
                    force_input=True,
                ),
                io.String.Input(
                    "text",
                    display_name="text",
                    multiline=True,
                    default="",
                ),
            ],
            outputs=[
                io.String.Output(
                    "out_response",
                    display_name="response",
                    tooltip="Cleaned Gemini response",
                ),
            ],
        )

    @classmethod
    def execute(self, response, text):
        
        response = str(response)
        response = response.replace("*", "")
        response = response.replace("#", "")
       
        return {
        "ui": {"response": [response]}, 
        "result": (response,)
        }
        
#################################################

class GeminiMarkdown(io.ComfyNode):

    DESCRIPTION = "Displays markdown text."

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiMarkdown",
            display_name=node_prefix() + " Artha Gemini Markdown",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "response",
                    display_name="markdown",
                    force_input=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    "out_response",
                    display_name="response",
                ),
            ],
        )

    @classmethod
    def execute(cls, response):
        
        response = str(response)

        return {
        "ui": {"response": [response]}, 
        "result": (response,)
        }
                       
#################################################

class GeminiInstruct(io.ComfyNode):

    DESCRIPTION = (
        "Gemini Instruct Node provides system-level instructions "
        "for other Gemini nodes."
    )

    ARTHA_INSTRUCT = io.Custom("ARTHAINSTRUCT")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ArthaGeminiInstruct",
            display_name=node_prefix() + " Artha Gemini Instruct",
            category=main_cetegory() + "/LLM/Gemini",
            description=cls.DESCRIPTION,
            inputs=[
                io.String.Input(
                    "role",
                    multiline=True,
                    default="You are an intelligent AI assistant.",
                ),
                io.String.Input(
                    "task",
                    multiline=True,
                    default="Your task is to...",
                ),
                io.String.Input(
                    "instructions",
                    multiline=True,
                    default="",
                ),
            ],
            outputs=[
                cls.ARTHA_INSTRUCT.Output(
                    "system_instruction",
                    display_name="system instruction",
                ),
            ],
        )

    @classmethod
    def execute(cls, role, task, instructions):

        role = str(role).replace('"', "")
        task = str(task).replace('"', "")
        instructions = str(instructions).replace('"', "")

        system_instruction = (
            f"Role: {role}\n\n"
            f"Task: {task}\n\n"
            f"{instructions}"
        )

        return (system_instruction,)