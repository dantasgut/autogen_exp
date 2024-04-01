import random
from types import SimpleNamespace

class DummyModelClient:
    """
    A non-interactive dummy client to use with AutoGen.
    Use when testing features that don't require a live LLM connection e.g. UI changes.
    Based on code from https://microsoft.github.io/autogen/blog/2024/01/26/Custom-Models/
    """
    def __init__(self, config, **kwargs):
        print(f"DummyModelClient config: {config}")
        
    def create(self, params):
        num_of_responses = params.get("n", 1)
        
        # Create a dummy data response class
        # adhering to AutoGen's ModelClientResponseProtocol
        response = SimpleNamespace()
        response.choices = []
        response.model = "dummy-model-client"
        
        for _ in range(num_of_responses):
            text = random.choice(
                [
                    "What's your favorite thing about space? Mine is space.",
                    "Space space wanna go to space yes please space. Space space. Go to space.",
                    "Atmosphere. Black holes. Astronauts. Nebulas. Jupiter. The Big Dipper.",
                    "Ohhh, the Sun. I'm gonna meet the Sun. Oh no! What'll I say? 'Hi! Hi, Sun!' Oh, boy!",
                    "Look, an eclipse! No. Don't look."
                ]
            )
            choice = SimpleNamespace()
            choice.message = SimpleNamespace()
            choice.message.content = text
            choice.message.function_call = None
            response.choices.append(choice)
        return response
        
    def message_retrieval(self, response):
        choices = response.choices
        return [choice.message.content for choice in choices]
        
    def cost(self, response) -> float:
        response.cost = 0
        return 0
        
    @staticmethod
    def get_usage(response):
        return {}