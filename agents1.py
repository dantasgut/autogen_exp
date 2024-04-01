import os
import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import DockerCommandLineCodeExecutor
from client import CustomModelClient



config_list = autogen.config_list_from_json(
    "CONFIG_LIST",
    filter_dict={
        "model": ["google/flan-t5-small"],
    },
)

llm_config = {"config_list": config_list, "cache_seed": 42}

#config_list = [{"model":"gpt-4", "api_key":os.environ["OPEN_API_KEY"]}]
#config_list = [{"model":"google/flan-t5-small", "api_key": os.environ["API_KEY"]}]

# Create an AssistantAgent instance named "assistant" with the LLM configuration.
assistant = AssistantAgent(name="assistant", system_message="You are agent spetialist in languages", llm_config=llm_config)


assistant.register_model_client(model_client_cls=CustomModelClient)

# Create an UserProxyAgent instance named "user_proxy" with code execution on docker.
code_executor = DockerCommandLineCodeExecutor(container_name="proxy_user", work_dir=".")
user_proxy = UserProxyAgent(name="user_proxy", code_execution_config={"executor":code_executor})

# The assistant receives a message from the user which contains the task description
user_proxy.initiate_chat(
    assistant,
    message = """
        Translate all messages I send in german
    """
)