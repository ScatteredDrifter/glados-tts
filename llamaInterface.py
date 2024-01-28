# this file acts as interface 
# for communicating with a local-first 
# llama LLM server

# --- /
# -- / internal imports
from glados import Glados,play_sound

# --- / 
# -- / external imports 
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

# --- / 
# -- / constants 

# server address follows the format of "http://<Your api-server IP>:port"
SERVER_ADRESS:str = "http://localhost:8080/v1" 

# denotes api key for llm used
# WARNING llama does not require any
API_KEY:str  = "sk-no-key-required" 

## registering user accordingly
# FIXME refactor into function
client = OpenAI(
    base_url= SERVER_ADRESS,
    api_key = API_KEY
)
# completion = client.chat.completions.create(
#     model="LLaMA_CPP",
#     messages=[
#         {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
#         {"role": "user", "content": "Write a limerick about python exceptions"}
#     ]
# )

# function to send message to llama
def request_response(message:str) -> str | None:
    completion = client.chat.completions.create(
        model="LLaMA_CPP",
        messages=[
            {"role": "system", "content": "You are EvilLyn, a good friend.You are very kind, helpful, and trying to be a pleasant friend. You also make nerdy jokes"},
            {"role": "user", "content": message}
        ]
    )
    received_completion:ChatCompletionMessage = completion.choices[0].message
    extracted_message = received_completion.content
    return extracted_message

# prettify printed messages 

def print_pretty(message:str,user:str) -> None:
    print("{} : {}\n".format(user,message))

if __name__ == "__main__":
    print("executing main function")
    glados_tts = Glados()
    glados_tts.load_glados_model()
    while True: 
        try:
            user_input:str = str(input("send your message to llama:\n"))
        except:
            print("invalid input, try again")
            continue
        received_response:str = request_response(user_input)
        print_pretty(received_response,"llama")
        # also playing with glados tts 
        temporary_file = glados_tts.generate_tts(received_response)
        play_sound(temporary_file)