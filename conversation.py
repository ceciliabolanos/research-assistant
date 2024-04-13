from rich.console import Console
from rich.markdown import Markdown
import json
from openai import OpenAI


GPT_MODEL = "gpt-4-turbo"

def chat_completion_request(client, messages, tools=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
def chat_completion_with_function_execution(client, messages, searcher, tools=[None]):
    """This function makes a ChatCompletion API call with the option of adding functions"""
    response = chat_completion_request(client, messages, tools)
    print(response)
    full_message = response.choices[0]
    if full_message.finish_reason == "tool_calls":
        print(f"Function generation requested, calling function")
        return call_searcher_function(client, messages, searcher, full_message, tools)
    else:
        print(f"Function not required, responding to user")
        return response

def call_searcher_function(client, messages, searcher, full_message, tools):
    """Function calling function which executes function calls when the model believes it is necessary.
    Currently extended by adding clauses to this if statement."""  
    if full_message.message.tool_calls[0].function.name == "similarity_search":
        try:
            parsed_output = json.loads(
                full_message.message.tool_calls[0].function.arguments
            )
            print("Getting search results")
            results = searcher.similarity_search(parsed_output["query"], parsed_output["k"])
            formatted_results = json.dumps(results)
            # results = search_by_keywords(parsed_output["query"], parsed_output["k"], parsed_output["filters"])
        except Exception as e:
            print(parsed_output)
            print(f"Function execution failed")
            print(f"Error message: {e}")
         
        messages.append(
            {
                "role": "function",
                "name": full_message.message.tool_calls[0].function.name,
                # "content": str(results),
                "content": formatted_results,
            }
        )
        try:
            print("Got search results, summarizing content")
            response = chat_completion_request(client=client, messages=messages, tools=tools)
            return response
        except Exception as e:
            print(type(e))
            raise Exception("Function chat request failed")
    else:
        raise Exception("Function does not exist and cannot be called") 
    
class Conversation:
    def __init__(self, tools, searcher):
        self.conversation_history = []
        self.client = OpenAI(api_key = OPENAI_API_KEY)
        self.tools = tools
        self.console = Console()  # Create a Rich console instance
        self.searcher = searcher

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def respond(self):
      chat_response = chat_completion_with_function_execution(
                      self.client,
                      self.conversation_history,
                      tools = self.tools,
                      searcher = self.searcher)
      assistant_message = chat_response.choices[0].message.content
      self.add_message("assistant", assistant_message)
      return assistant_message
      # display(Markdown(assistant_message))


    def chat(self):
        self.console.print("Chat started. Type 'exit' to quit.", style="bold yellow")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                self.console.print("Exiting chat.", style="bold red")
                break
            self.add_message("user", user_input)

            # Generate the assistant's response and display it
            response = self.respond()
            self.console.print(f"Bot: {response}", style="bold green")
            self.add_message("assistant", response)

           
