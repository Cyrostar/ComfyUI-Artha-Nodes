from .math.math import MathIntegerNode
from .math.math import MathFloatNode
from .math.math import MathI2FNode
from .math.math import MathF2INode
from .math.math import MathOperationNode
from .math.math import MathRandomNode
from .math.math import MathNumberNode

from .logic.logic import LogicBasicNode
from .logic.logic import LogicGateNode
from .logic.logic import LogicCompareNode
from .logic.logic import LogicPassthruNode
from .logic.logic import LogicPrintNode

from .project.project import ProjectSetupNode
from .project.project import ProjectPrefixNode
from .project.project import ProjectSeedNode

from .flow.flow import FlowInputSelector
from .flow.flow import FlowPause

from .image.image import ImageTransformNode
from .image.image import ImageColorMatchNode
from .image.image import ImagePreviewNode
from .image.image import ImageSaveNode

from .llm.gemini import GeminiQuestion
from .llm.gemini import GeminiOperation
from .llm.gemini import GeminiTranslate
from .llm.gemini import GeminiImagen
from .llm.gemini import GeminiSpeech
from .llm.gemini import GeminiVision
from .llm.gemini import GeminiMotion
from .llm.gemini import GeminiPrompter
from .llm.gemini import GeminiCondense
from .llm.gemini import GeminiPortrait
from .llm.gemini import GeminiFace
from .llm.gemini import GeminiBody
from .llm.gemini import GeminiForm
from .llm.gemini import GeminiCloth
from .llm.gemini import GeminiMakeup
from .llm.gemini import GeminiCompose
from .llm.gemini import GeminiSubject
from .llm.gemini import GeminiScenery
from .llm.gemini import GeminiCamera
from .llm.gemini import GeminiLight
from .llm.gemini import GeminiStyle
from .llm.gemini import GeminiResponse
from .llm.gemini import GeminiMarkdown
from .llm.gemini import GeminiInstruct

NODES = [
    MathIntegerNode,
    MathFloatNode,
    MathI2FNode,
    MathF2INode,
    MathOperationNode,
    MathRandomNode,
    MathNumberNode,
    LogicBasicNode,
    LogicGateNode,
    LogicCompareNode,
    LogicPassthruNode,
    LogicPrintNode,
    ProjectSetupNode,
    ProjectPrefixNode,
    ProjectSeedNode,
    FlowInputSelector,
    FlowPause,
    ImageTransformNode,
    ImageColorMatchNode,
    ImagePreviewNode,
    ImageSaveNode,
    GeminiQuestion,
    GeminiOperation,
    GeminiTranslate,
    GeminiImagen,
    GeminiSpeech,
    GeminiVision,
    GeminiMotion,
    GeminiPrompter,
    GeminiCondense,
    GeminiPortrait,
    GeminiFace,
    GeminiBody,
    GeminiForm,
    GeminiCloth,
    GeminiMakeup,
    GeminiCompose,
    GeminiSubject,
    GeminiScenery,
    GeminiCamera,
    GeminiLight,
    GeminiStyle,
    GeminiResponse,
    GeminiMarkdown,
    GeminiInstruct,
]