import math
from types import SimpleNamespace
from typing import Optional, Type

import autogen

from autogen import AssistantAgent, UserProxyAgent
from transformers import AutoTokenizer, GenerationConfig, T5ForConditionalGeneration, AutoModelForCausalLM, AutoModel, T5Config

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool


class CustomModelClient:
    def __init__(self, config, **kwargs):        
        self.device = config.get("device", config["device"])
        print(config)

        '''config_model = T5Config(
            vocab_size= 32128,
            d_model= 1024,
            num_layers= 6,
            num_heads= 8,
            dropout_rate= 0.1,
            max_position_embeddings= 1024,
            initializer_factor= 1.0,
            eos_token_id= 1,
            pad_token_id= 0,
            bos_token_id= 1
        )'''
        
        
        #self.model = AutoModelForCausalLM.from_pretrained(config["model"], config = config_model)
        
        self.model = T5ForConditionalGeneration.from_pretrained(config["model"]).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = config["model"], legacy=False, use_fast = False)

        self.model_name = config["model"]
        
        
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # params are set by the user and consumed by the user since they are providing a custom model
        # so anything can be done here
        gen_config_params = config.get("params", {})
        self.max_length = gen_config_params.get("max_length", 256)

    def create(self, params):
        if params.get("stream", False) and "messages" in params:
            raise NotImplementedError("Local models do not support streaming.")
        else:
            num_of_responses = params.get("n", 1)

            # can create my own data response class
            # here using SimpleNamespace for simplicity
            # as long as it adheres to the ClientResponseProtocol

            response = SimpleNamespace()

            inputs = self.tokenizer.apply_chat_template(
                conversation = params["messages"], 
                return_tensors="pt", 
                add_generation_prompt=True
            ).to(self.device)
            inputs_length = inputs.shape[-1]

            # add inputs_length to max_length
            max_length = self.max_length + inputs_length
            print("Quantidade máxima",self.max_length)
            generation_config = GenerationConfig(
                max_length = max_length,
                eos_token_id = self.tokenizer.eos_token_id,
                pad_token_id = self.tokenizer.pad_token_id,
            )

            response.choices = []
            response.model = self.model_name
            print(self.model.device)
            print(inputs.device)
  
            for _ in range(num_of_responses):
                outputs = self.model.generate(
                    inputs = inputs, 
                    generation_config=generation_config
                    )
                # Decode only the newly generated text, excluding the prompt
                text = self.tokenizer.decode(token_ids = outputs[0, inputs_length:])
                choice = SimpleNamespace()
                choice.message = SimpleNamespace()
                choice.message.content = text
                choice.message.function_call = None
                response.choices.append(choice)

            return response

    def message_retrieval(self, response):
        """Retrieve the messages from the response."""
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        """Calculate the cost of the response."""
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        # returns a dict of prompt_tokens, completion_tokens, total_tokens, cost, model
        # if usage needs to be tracked, else None
        return {}


class CustomToolInput(BaseModel):
    income: float = Field()


class CustomTool(BaseTool):
    name = "tax_calculator"
    description = "Use this tool when you need to calculate the tax using the income"
    args_schema: Type[BaseModel] = CustomToolInput

    def _run(self, income: float):
        return float(income) * math.pi / 100


# Define a function to generate llm_config from a LangChain tool
def generate_llm_config(tool):
    # Define the function schema based on the tool's args_schema
    function_schema = {
        "name": tool.name.lower().replace(" ", "_"),
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    if tool.args is not None:
        function_schema["parameters"]["properties"] = tool.args

    return function_schema