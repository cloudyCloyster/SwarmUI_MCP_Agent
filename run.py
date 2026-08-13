from sre_parse import CATEGORIES
from typing import TypedDict, Annotated, Literal
import operator
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tools import generate_image # Import the generate_image tool from tools.py
import re
import json
import yaml
import os
import streamlit as st
import requests
import base64
import time

def make_spinner(text = "In progress..."):
    with st.spinner(text):
        yield
sp = iter(make_spinner())

def extract_base64_from_data_uri(data_uri: str) -> str:
    # Extract base64 part from data URI
    match = re.match(r"data:.*?;base64,(.*)", data_uri)
    if not match:
        raise ValueError("Invalid data URI format")
    return match.group(1)

def get_image_data_uri(img_url: str, api_base_url: str) -> str:
    if img_url.startswith("data:"):
        return img_url  # Already a data URI
    # Otherwise, fetch the image from the API
    # #Ensure the URL is absolute
    if not img_url.startswith("http"):
        img_url = api_base_url.rstrip("/") + "/" + img_url.lstrip("/")
    resp = requests.get(img_url)
    resp.raise_for_status()
    mime = resp.headers.get("Content-Type", "image/png")  # Default to PNG
    b64 = base64.b64encode(resp.content).decode("utf-8")
    return f"data:{mime};base64,{b64}"

#Define agent state structure
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_input: str
    needs_image_generation: bool
    image_params: dict
    enhanced_prompt: str
    image_type_category: str
    explicit_prompt_type: str #'none', 'explicit', 'enhance'

class SwarmUIAgent:
    def __init__(self, model_name: str = "dolphin-mistral"):
        st.title("SwarmUI Agent")
        with st.expander("Settings", expanded=False):
            self.port = st.text_input("SwarmUI API Port", value="7801")
            self.llm_model = st.selectbox("LLM Model",["dolphin-mistral", "dolphin-llama3:8b", "huihui_ai/dolphin3-abliterated:latest", "sam860/dolphin3-qwen2.5:3b", "hammerai/mistral-nemo-uncensored"])
            self.vision_model = st.selectbox("Vision Model", ["None", "gemma3:12b", "llava","bakllava","qwen-vl"])
            self.tag_model = st.selectbox("Tag Model", ["None", "DanTagGen"])
        self.llm = OllamaLLM(model=self.llm_model)
        if self.llm_model == "dolphin-llama3:8b":
            self.llm.num_ctx = 256000;
        self.tag_llm = OllamaLLM(model=self.tag_model if self.tag_model != "None" else self.llm_model)
        self.vision_llm = OllamaLLM(model=self.vision_model)
        self.tools = [generate_image]
        self.graph = self._build_graph()
        self.categories = self.load_image_generation_categories()
        self.output = st.empty()
        self.loading = st.empty()
        self.input = st.text_area("User Input",
                                   help="Enter your request to the agent. Mention 'generate' to trigger SwarmUI image generation.",
                                   placeholder="Enter your request for the agent. Mention 'generate' for SwarmUI image generation.")
        self.submit = st.button("Submit")
        if "welcome_shown" not in st.session_state:
            with self.output.container():
                self.output.write(
                    f"""
                    Hello! I'm an agent that can help you generate images using your local SwarmUI API\n
                    **Setting Categories**: NONE, {", ".join(list(self.get_available_categories().keys())[1:])}\n
                    **Example**: Generate an anime image of a character with blue hair
                    """
                )

        if self.submit:
            try:
                st.session_state["welcome_shown"] = True
                result = self.run(self.input)
                # Accumulate all AI messages as a single string, separated by double newlines for clarity
                all_msgs = []
                for msg in result['messages']:
                    if isinstance(msg, AIMessage):
                        all_msgs.append(msg.content)
                
                # Join all messages and display with st.markdown for multiline support
                self.output.markdown('\n\n'.join(all_msgs).replace('\n', '  \n'))
            except Exception as e:
                self.output.write(f"Error: {e}")

    def load_image_generation_categories(self, config_path: str = "image_generation_categories.yaml") -> dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)["categories"]

    def _build_graph(self):
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_input", self.analyze_input)
        workflow.add_node("natural_response", self.natural_response)
        workflow.add_node("enhance_prompt", self.enhance_prompt)
        workflow.add_node("determine_parameters", self.determine_parameters)
        workflow.add_node("prepare_tool_call", self.prepare_tool_call)
        workflow.add_node("execute_tool", ToolNode(self.tools))
        workflow.add_node("format_tool_response", self.format_tool_response)

        # Define entry
        workflow.set_entry_point("analyze_input")

        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_input",
            self.route_response,
            {
                "natural":"natural_response",
                "generate_image":"enhance_prompt"
            }
        )

        # Add paths to end
        #Natural response path
        workflow.add_edge("natural_response", END)

        #Image generation path
        workflow.add_edge("enhance_prompt", "determine_parameters")
        workflow.add_edge("determine_parameters", "prepare_tool_call")
        workflow.add_edge("prepare_tool_call", "execute_tool")
        workflow.add_edge("execute_tool", "format_tool_response")
        workflow.add_edge("format_tool_response", END)

        return workflow.compile()

    def analyze_input(self, state: AgentState) -> AgentState:
        """Analyze user input to determine if image generation is needed"""
        user_input = state["user_input"].lower()
        # this starts the spinning
        next(sp)

        input_category = self.classify_input_llm(user_input).strip('"').strip('`').strip();
        needs_image = input_category == "image_generation"

        # Extract explicit parameters from user input (these will override presets)
        explicit_params, explicit_prompt_type = self.extract_explicit_params(state["user_input"])

        return {
            **state,
            "needs_image_generation": needs_image,
            "image_params": explicit_params,
            "enhanced_prompt": "",
            "image_type_category": "",
            "explicit_prompt_type": explicit_prompt_type
        }

    image_generation_category_examples = [
        "image_generation Category examples:\n",
        "User input: \"Generate an image of a cat in a spacesuit.\"",
        "Category: image_generation\n",
        "User input: \"Can you make a drawing of a futuristic city?\"",
        "Category: image_generation\n",
        "User input: \"Create a stylized heart graphic\"",
        "Category: image_generation\n",
        "User input: \"Make beautiful outdoor art\"",
        "Category: image_generation\n",
        "User input: \"Draw a portrait of a girl on a beach\"",
        "Category: image_generation\n",
    ]
    general_input_category_examples = [
        "general_input Category examples:",
        "User input: \"What's the weather like today?\"",
        "Category: general_input\n",
        "User input: \"Tell me a joke.\"",
        "Category: general_input\n",
        "User input: \"Generate a beautiful story about a dragon and a princess\"",
        "Category: general_input\n",
        "User input: \"Make a recipe for soup\"",
        "Category: general_input\n",
    ]
    
    def classify_input_llm(self, user_input: str) -> str:
        categories = self.image_generation_category_examples+self.general_input_category_examples
        categories_str = "\n".join(categories)
        prompt = (
            "Classify the following user input\n\n"
            f"{categories_str}"
            f"User input: \"{user_input}\"\n"
            "Category: "
        )
        response = self.llm.invoke(prompt).strip().split()[0]  # Take the first word as the label
        # Fallback to 'general_input' if not recognized
        return response

    def extract_explicit_params(self, user_input: str) -> dict:
        """Extract explicitly specified image generation parameters from user input"""
        params = {}
        explicit_prompt_type = "none"
        # Extract numerical parameters
        param_patterns = {
            'steps': r'steps?\s*:?\s*(\d+)',
            'cfgScale': r'cfg(?:\s*scale)?\s*:?\s*([\d.]+)',
            'seed': r'seed\s*:?\s*(-?\d+)',
            'images': r'images?\s*:?\s*(\d+)',
            'width': r'width\s*:?\s*(\d+)',
            'height': r'height\s*:?\s*(\d+)',
            'refinerControlPercentage': r'refinerControlPercentage?\s*:?\s*(\d+)',
            'refinerUpscale': r'refinerUpscale?\s*:?\s*(\d+)',
        }
        
        for param, pattern in param_patterns.items():
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                value = match.group(1)
                if param in ['cfgScale', 'refinerControlPercentage', 'refinerUpscale']:
                    params[param] = float(value)
                else:
                    params[param] = int(value)
        
        # Extract string parameters
        string_patterns = {
            'prompt': r'prompt\s*:?\s*["\']([^"\']+)["\']',
            'negative': r'negative\s*:?\s*["\']([^"\']+)["\']',
            'model': r'model\s*:?\s*["\']?([^"\']+)["\']?',
            'sampler': r'sampler\s*:?\s*["\']?([^"\']+)["\']?',
            'scheduler': r'scheduler\s*:?\s*["\']?([^"\']+)["\']?',
        }
        
        for param, pattern in string_patterns.items():
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                params[param] = match.group(1).strip()

        prompt = None
        match = re.search(r'prompt\s*:?\s*["\']([^"\']+)["\']', user_input, re.IGNORECASE)
        if match:
            prompt = match.group(1).strip()
            params['prompt'] = prompt
            # Detect if prompt is already enhanced
            if prompt.lower().startswith("enhance:"):
                explicit_prompt_type = "enhance"
            else:
                explicit_prompt_type = "explicit"

        return params, explicit_prompt_type

    def route_response(self, state: AgentState) -> Literal["natural", "generate_image"]:
        """Route to appropriate response type based on analysis"""
        return "generate_image" if state["needs_image_generation"] else "natural"

    def enhance_prompt(self, state: AgentState) -> AgentState:
        """Enhance the user's prompt for better image generation"""
        if state.get("explicit_prompt_type") == "explicit":
            # Use the explicit prompt as the enhanced prompt
            return {
                **state,
                "enhanced_prompt": state["image_params"].get("prompt", "")
            }

        user_input = state["user_input"]
        output_parser = CommaSeparatedListOutputParser() # Assuming you're using this from previous advice
        format_instructions = output_parser.get_format_instructions()
        
        if self.tag_model == "DanTagGen":
            enhancement_prompt_template = f"""Generate tags the precisely describe {user_input}"""
        else:
            enhancement_prompt_template = f"""Generate tags that describe the following input: {user_input}\n\n
            {format_instructions}\n
            Guidelines: \n
            ONLY INCLUDE: tags, tag phrases, descriptive phrases\n
            DO NOT INCLUDE: notes, explanations or other prose, tags or details contrary or in opposition to the input.\n
            DO INCLUDE: additional detail tags similar to the input, styles and image quality tags that enhance the input\n 
            HIGHLY ENCOURAGED: random tags for character details, setting, environment, mood, emotion, and actions when missing\n
            AVOID: Beginning with tags:, enhanced:, prompt:\n"""

        final_enhancement_prompt = enhancement_prompt_template.format(user_input=user_input)
        raw_llm_output = self.tag_llm.invoke(final_enhancement_prompt).strip()
        cleaned_llm_output = raw_llm_output # Start with the raw output
        enhanced_prompt = ""
        # Check if the active LLM is DanTagGen, as this cleaning is specific to its format
        if self.tag_model == "DanTagGen":
            # Extract the content of the 'general:' label
            general_match = re.search(r"general:\s*(.*)", cleaned_llm_output, re.IGNORECASE)

            if general_match:
                general_content = general_match.group(1)
            else:
                # Fallback if 'general:' label is not found
                self.output.write(
                    f"⚠️ **Warning**: 'general:' label not found in DanTagGen output. "
                    f"The specialized cleaning might not work as expected. Raw output: \"{raw_llm_output}\""
                )
                general_content = ""

            # Remove special tokens by finding "<|", then any characters NOT a "|", then "|>", followed by optional whitespace
            if general_content:
                cleaned_tags_intermediate = re.sub(r"<\|[^\|]+\|>\s*", "", general_content)
            else:
                cleaned_tags_intermediate = ""

            if cleaned_tags_intermediate:
                # Replace multiple consecutive spaces with a single space
                temp_cleaned_tags = re.sub(r'\s+', ' ', cleaned_tags_intermediate)
                # Normalize spacing around commas (e.g., "tag1  ,  tag2" -> "tag1, tag2")
                temp_cleaned_tags = re.sub(r'\s*,\s*', ', ', temp_cleaned_tags)
                # Remove any leading or trailing commas and spaces
                enhanced_prompt = temp_cleaned_tags.strip(' ,')

        if not enhanced_prompt:
            try:
                # Attempt to parse the output into a list of strings
                parsed_tags_list = output_parser.parse(cleaned_llm_output)
                # Join the list back into a comma-separated string.
                # This intrinsically ensures no quotes around individual elements.
                # Also, strip individual tags of any lingering whitespace or quotes missed by the parser
                # (though a good parser should handle this).
                cleaned_tags = [tag.strip().strip('\'"') for tag in parsed_tags_list]
                enhanced_prompt = ", ".join(filter(None, cleaned_tags)) # Filter out empty strings
            except Exception as e:
                # Fallback if parsing fails (e.g., LLM output is too malformed)
                self.output.write(f"⚠️ Warning: Output parser failed for enhanced prompt. Error: {e}. Using basic cleaning on raw output: \"{raw_llm_output}\"")
                # Basic cleaning: aggressively remove quotes and re-join
                # Remove all standalone quotes that might be wrapping tags
                temp_prompt = re.sub(r'["\']([^"\']+)["\']', r'\1', cleaned_llm_output) # "tag" -> tag
                temp_prompt = re.sub(r'(?<!\w)["\']|["\'](?!\w)', '', temp_prompt) # Remove remaining loose quotes
                # Normalize spacing around commas
                tags = [tag.strip() for tag in temp_prompt.split(',')]
                enhanced_prompt = ", ".join(filter(None, tags)) # Filter out any empty strings from multiple commas etc.
        
            # Ensure no leading/trailing commas or excessive internal spacing
            enhanced_prompt = re.sub(r'\s*,\s*', ', ', enhanced_prompt).strip(', ')

        self.output.write(
            f"""\n
            Generating image with SwarmUI...\n\n
            📝 **Base Prompt**: {user_input}\n
            ✨ **Enhanced Prompt**: {enhanced_prompt}
            """
        )

        return {
            **state,
            "enhanced_prompt": enhanced_prompt
        }

    def determine_parameters(self, state: AgentState) -> AgentState:
        """Determine optimal image generation parameters based on the prompt and request type"""
        user_input = state["user_input"].lower()
        enhanced_prompt = state["enhanced_prompt"].lower()
        
        # Reload categories from YAML each time
        self.categories = self.load_image_generation_categories()

        detected_category = None
        for category, data in self.categories.items():
            if any(keyword in user_input for keyword in data.get("keywords", [])):
                detected_category = category
                break
        if not detected_category:
            detected_category = next(iter(self.categories))  # fallback to first category

        base_params = self.categories[detected_category]["parameters"].copy()
        
        # Override with any explicitly specified parameters
        final_params = {**base_params, **state["image_params"]}
        
        # Set the enhanced prompt as the main prompt
        final_params["prompt"] = state["enhanced_prompt"]
        
        return {
            **state,
            "image_params": final_params,
            "image_type_category": detected_category
        }

    def natural_response(self, state: AgentState) -> AgentState:
        """Generate a natural language response"""
        # Create a simple prompt for natural conversation
        prompt = f"Respond naturally to this user input: {state['user_input']}"
        
        response = self.llm.invoke(prompt)
        next(sp, None)

        return {
            **state,
            "messages": [AIMessage(content=response)]
        }

    def prepare_tool_call(self, state: AgentState) -> AgentState:
        """Prepare the tool call message for execution"""        
        tool_call = {
            "name": "generate_image",
            "args": state["image_params"],
            "id": "generate_image_call_1"
        }
        
        ai_message = AIMessage(
            content="",
            tool_calls=[tool_call]
        )
        
        return {
            **state,
            "messages": [ai_message]
        }

    def format_tool_response(self, state: AgentState) -> AgentState:
        """Format the tool response for display"""
        # Get the last message (should be a ToolMessage from the tool execution)
        last_message = state["messages"][-1]
        api_base_url = f"http://localhost:{self.port}"  # Change to your SwarmUI API base URL

        if isinstance(last_message, ToolMessage):
            # The tool has been executed, now format the response
            try:
                # Parse the tool result
                result = json.loads(last_message.content) if isinstance(last_message.content, str) else last_message.content
                
                # Create a user-friendly response
                if 'images' in result and result['images']:
                    response = f"✅ **Image generated successfully!**\n\n"
                    response += f"✨ **User Request**: {state['user_input']}\n"
                    response += f"🎨 {'**Enhanced Prompt**' if state['explicit_prompt_type'] != 'explicit' else '**Prompt**'}: {state['enhanced_prompt']}\n"
                    response += f"🎯 **Style**: {state['image_type_category']}\n"
                    response += f"📸 **Generated** {len(result['images'])} image(s)\n"
                    
                    # Show key parameters used
                    params = state['image_params']
                    response += f"⚙️ **Parameters**: *{params.get('model','INVALID MODEL')}*, "
                    response += f"**{params.get('width', '-')}x{params.get('height', '-')}**, "
                    response += f"**Steps**: {params.get('steps', 'INVALID STEPS')}, **CFG**: {params.get('cfgScale', 'INVALID CFG')}\n"
                    
                    for i, img_url in enumerate(result['images'], 1):
                        data_uri = get_image_data_uri(img_url, api_base_url)
                        st.image(data_uri, caption=f"Image {i}")
                        response += f"🔗 **Image {i}**: {img_url}\n"

                        if self.vision_model != "None":
                            # Extract base64 string for vision model
                            base64_str = extract_base64_from_data_uri(data_uri)
                            description_prompt = "Describe this image. If a detail is too small or blurry to clearly understand, ignore it."
                            vision_response = self.vision_llm.invoke(
                                input=description_prompt,
                                images=[base64_str]
                            )
                            response += f"📝 **Description**: {vision_response}\n"
                            # spinning end
                            next(sp, None)
                        else:
                            # spinning end
                            next(sp, None)

                else:
                    response = "❌ Image generation failed - no images returned"
                    
            except Exception as e:
                response = f"❌ Error processing image generation result: {str(e)}"
        else:
            response = "❌ Unexpected response format from image generation tool"
        
        return {
            **state,
            "messages": [AIMessage(content=response)]
        }

    def run(self, user_input: str) -> dict:
        """Run the agent with user input"""
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_input": user_input,
            "needs_image_generation": False,
            "image_params": {},
            "enhanced_prompt": "",
            "image_type_category": ""
        }
        
        result = self.graph.invoke(initial_state)
        return result

    def get_available_categories(self) -> list:
        """Return list of available image categories and their descriptions"""
        return self.categories

if __name__ == "__main__":
    agent = SwarmUIAgent()
