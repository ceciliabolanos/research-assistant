from rich.console import Console
from rich.markdown import Markdown
import json
from openai import OpenAI
#from calling_mistral import mistral_process_nl_query


class Conversation:
    """
    Manages conversations using an AI model and dispatches tools as needed.
    
    Attributes:
        conversation_history (list): A list to store conversation history.
        client (OpenAI): The OpenAI client initialized with an API key.
        tool_dispatcher (ToolDispatcher): A dispatcher to handle tool-specific operations.
        console (Console): Console object for user interaction.
    """
    def __init__(self, api_key, searcher, tools, mistral_option = 'no', chat_model='gpt-3.5-turbo-0125'):
        self.conversation_history = []
        self.client = OpenAI(api_key=api_key)
        self.searcher = searcher
        self.console = Console()
        self.tools = tools
        self.mistral = mistral_option
        self.chat_model = chat_model

    def add_message(self, role, content):
        """Adds a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def chat_completion_request(self, messages):
        """
        Requests a chat completion from the AI model.

        Args:
            messages (list): The current conversation history.
            model (str): The model identifier.

        Returns:
            Completion: The response from the AI model.
        """
        try:
            response = self.client.chat.completions.create(model=self.chat_model, messages=messages,tools=self.tools)
            return response
        except Exception as e:
            self.console.print(f"Unable to generate ChatCompletion response: {str(e)}", style="bold red")
            return None
        

    def respond(self):
        response = self.chat_completion_request(self.conversation_history)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        available_functions = {
            "similarity_search": self.searcher.similarity_search,  # Assuming searcher is correctly initialized
        }
        
        if tool_calls:
            self.conversation_history.append(response_message)  
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                # query = mistral_process_nl_query(function_args.get("query")) if self.mistral == 'yes' else function_args.get("query"),
                    query = function_args.get("query"),
                    k=function_args.get("k"),
                )
                # Format the response into a string suitable for the conversation history
                formatted_response = json.dumps(function_response, indent=2)
                self.conversation_history.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": formatted_response,
                    }
                )  
                
        try:
            second_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=self.conversation_history,
                tools = self.tools
            )   
            self.conversation_history.append({"role": "assistant", "content": second_response.choices[0].message.content})
            return second_response.choices[0].message.content
        except Exception as e:
            self.console.print(f"Unable to generate ChatCompletion response: {str(e)}", style="bold red")
            return None


    def chat(self):
        self.console.print("Chat started. Type 'exit' to quit.", style="bold yellow")
        while True:
            user_input = self.console.input("You: ")
            if user_input.lower() == "exit":
                self.console.print("Exiting chat.", style="bold red")
                break
            self.add_message("user", user_input)
            response = self.respond()
            self.console.print(f"Response: {response}", style="bold green")
