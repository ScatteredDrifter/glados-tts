import sys
import os
import logging
import time
from typing import Optional
from flask import Flask, Response, request, send_file, after_this_request
import urllib.parse
from glados import Glados, play_sound,remove_audio_file

sys.path.insert(0, os.getcwd()+'/glados_tts')
logging.basicConfig(filename='glados_engine_service.log',
    format='[%(asctime)s.%(msecs)03d] [%(levelname)s]\t%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',level=logging.DEBUG)

def printed_log(message):
    logging.info(message)
    print(message)

def print_timelapse(processName,old_time):
    printed_log(f"{processName} took {str((time.time() - old_time) * 1000)} ms")

def stream_and_remove_file(file_path:str):
    with open(file_path,'rb') as file:
        yield from file
    remove_audio_file(file_path)

# If the script is run directly, assume remote engine
if __name__ == "__main__":
    # FIXME improve writing
    # FIXME remove global state
    printed_log("Initializing TTS Remote Engine...")
    glados:Glados = Glados()
    glados.load_glados_model()
    PORT:int = 8124

    printed_log("Initializing webserver")
    app = Flask(__name__)
    
    # listening for request that will synthesizes text
    # returns audiofile via http
    @app.route('/synthesize/', defaults={'text': ''},methods=["POST","GET"])
    @app.route('/synthesize/<path:text>',methods=["POST","GET"])
    def synthesize(text:str):

        # default value for synthesizing 
        # FIXME no primitive obsession please!
        input_text:str = text 
        
        # receive text from get-request
        if(request.method=="GET"):
            # receiving text as url encoded get request 
            input_text = request.args.get('text')
        
        # receive text from post-request
        elif(request.method=="POST"):
            input_text = request.data.decode('ascii')

        # aborting request in case nothing was provided 
        if input_text == "":
            return 
        
        printed_log(f"given text: {input_text}")
        # get audio file
        old_time:float = time.time()
        output_file = glados.generate_tts(input_text)
        print_timelapse("Time Generating audio file: ",old_time)
        
        # streaming file to client via generator 
        # to delete the file after the request was sent
        return_response = app.response_class(stream_and_remove_file(output_file),mimetype="audio/wav")
        return_response.headers["Content-Disposition"] = f"attachment; filename=glados_tts.wav"
        return return_response
    
    # --- / 
    # -- / also listening for requests that should be played locally on the server 
    # used for services sending a request they cannot speak themself 
    @app.route('/synthesize-local/', defaults={'text': ''},methods=["POST","GET"])
    def synthesize_and_speak(text:str):
        '''
        receives text and plays it locally
        @param text: String, input text to be synthesized

        '''

        
        if text == "": 
            # no explicit text provided
            return 
        # handling     
        if(request.method=="GET"):
            # receiving text as url encoded get request 
            input_text = request.args.get('text')
        elif(request.method=="POST"):
            input_text = request.data.decode('ascii')
        # logging request
        printed_log(f"Input text: {input_text}")
        output_file = glados.generate_tts(input_text)
        # playing sound locally
        play_sound(output_file)
        # removing file after it was played
        remove_audio_file(output_file)
        
    # --- /
    # -- / allowing to send promp and interact with lama interface via get-requests 
    @app.route('/ask_llama/', defaults={'query': ''},methods=["GET"])
    def synthesize_lama_response(query:str):
        '''
        #FIXME redundant with synthesize_and_speak!
        receives query and and plays it locally  
        '''
        if query == "": 
            # no explicit text provided
            return 
        # handling     
        if(request.method=="GET"):
            # receiving text as url encoded get request 
            # FIXME improve typing
            query:str = request.args.get('text')
        elif(request.method=="POST"):
            query:str = request.data.decode('ascii')
        # logging request
        printed_log(f"Input text: {input_text}")
        output_file = glados.generate_tts(input_text)
        # playing sound locally
        play_sound(output_file)
        
        
        
    
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    printed_log(f"Listening in http://localhost:{PORT}/synthesize/{'{PRHASE}'}")
    app.run(host="0.0.0.0", port=PORT)
