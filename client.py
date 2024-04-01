from types import SimpleNamespace


class CustomModelClient:
    def __init__(self, config, **kwargs):
        print(f"CustomModelClient config: {config}")

    def create(self, params):
        num_of_responses = params.get("n", 1)

        # can create my own data response class
        # here using SimpleNamespace for simplicity
        # as long as it adheres to the ModelClientResponseProtocol

        response = SimpleNamespace()
        response.choices = []
        response.model = "google/fla-t5-small" # should match the OAI_CONFIG_LIST registration

        for _ in range(num_of_responses):
            text = "this is a dummy text response"
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