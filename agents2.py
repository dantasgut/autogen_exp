import autogen
from autogen import AssistantAgent,UserProxyAgent
from dummy_client import DummyModelClient
from custom_client import CustomModelClient

config_list = autogen.config_list_from_json(
    "CONFIG_LIST",
    filter_dict={
        "model_client_cls": ["CustomModelClient"],
    },
)

llm_config = {"config_list": config_list, "cache_seed": 42}





agents = []

user = UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    human_input_mode="TERMINATE",
)
agents.append(user)


coder = AssistantAgent(
    name="Traductor_franch",
    llm_config=llm_config,
)
coder.register_model_client(model_client_cls=CustomModelClient)
agents.append(coder)

mn = AssistantAgent(
    name="Traductor_german",
    llm_config=llm_config,
)
mn.register_model_client(model_client_cls=CustomModelClient)
agents.append(mn)

'''for agent in agents:
    print(agent)
    if(not isinstance(agent, autogen.UserProxyAgent)):
        print(agent)
        agent.register_model_client()'''

groupchat = autogen.GroupChat(agents=agents, messages=[], max_round=5)
manager = autogen.GroupChatManager(groupchat=groupchat)

agents[0].initiate_chat(
    manager, message="Translate to franch and germany the frase: Happy easter for you and your familly"
)