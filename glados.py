# --- / 
# -- / File contains logic for processing text to speech
# -- | using torch and hifigan


# --- / 
# -- / external imports
import numpy
import numpy.typing
import torch
import random
import logging
from scipy.io.wavfile import write
import time
import os
from sys import modules as mod
import sys

from torch.jit import ScriptFunction, ScriptModule

try:
    import winsound
except ImportError:
    print("no winsound")
    from subprocess import call

# --- /
# -- / internal imports
from utils.tools import prepare_text

# --- / 
# -- / configuring logging module
logging.basicConfig(filename='glados_service.log',
    format='[%(asctime)s.%(msecs)03d] [%(levelname)s]\t%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',level=logging.DEBUG)

# FIXME remove global state
#Global variables
audio_path = os.getcwd()+'/audio/'

# --- / 
# -- / definition of Glados class 

class Glados:
    
    # FIXME improve writing
    glados_model:ScriptModule = None
    vocoder:ScriptFunction = None
    device:str = None


    def __init__(self):
        ''' 
        loads models and checks if audio folder exists
        '''
        check_audio_folder()

    def get_available_device(self,option_devices):
        '''
        selects available devices to run model on 
        options are: 
        - vulkan
        - cuda
        - cpu 
        '''
        if torch.is_vulkan_available() and ("vulkan" in option_devices):
            device_found = 'vulkan'
        elif torch.cuda.is_available() and  ("cuda" in option_devices):
            device_found = 'cuda'
        else:
            device_found = 'cpu'
        printed_log(f"Device selected: {device_found}.")
        return device_found

    def load_models(self):
        try:
            self.glados_model = torch.jit.load('models/glados.pt')
            self.vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=self.device)
        except Exception as exception:
            printed_log("could not load glados or vocoder")
            printed_log(exception)
            sys.exit()
        # preloading model with test-run 
        for i in range(4):
            init = self.glados_model.generate_jit(prepare_text(str(i)))
            init_mel = init['mel_post'].to(self.device)
            init_vo = self.vocoder(init_mel)
        printed_log(f"Models loaded.")

    def load_glados_model(self):
        option_devices = ["vulkan","cuda"]
        while(self.glados_model==None):
            self.device = self.get_available_device(option_devices)
            # try:
            self.load_models()
            # except:
            printed_log(f"Exception loading device "+self.device)
            # removing device from list of available devices
            #if(self.device != 'cpu'): option_devices.remove(self.device)

    def get_audio_from_text(self,text) -> numpy.typing.NDArray:
        '''
        #FIXME add type hint
        @param text: String, input text to be synthesized
        @return: numpy array, audio data 
        prepares text and pipes it through model
        '''
    	# Tokenize, clean and phonemize input text
        phonemized_text = prepare_text(text)
        with torch.no_grad():
            # Generate generic TTS-output
            old_time = time.time()
            tts_output = self.glados_model.generate_jit(phonemized_text)

            # Use HiFiGAN as vocoder to make output sound like GLaDOS
            mel = tts_output['mel_post']
            audio = self.vocoder(mel)
            print_timelapse("The audio sample: ",old_time)

            # Normalize audio to fit in wav-file
            audio = audio.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16')
        return audio

    # Generate audio file from given text as string
    # @return String, denoting path to saved file
    def generate_tts(self, input_text:str) -> str:
        '''
        #FIXME improve writing
        @param input_text: String, input text to be synthesized
        @return: String, denoting path to saved file
        generates audio file from given text, saves it to disk and returns path to file 
        '''
        filename = filename_parse(input_text)
        audio_file_exist = check_audio_file(filename)
        if (audio_file_exist):
            output_file:str = f"{audio_path}{filename}"
        else:
            audio = self.get_audio_from_text(input_text)
            output_file:str = save_audio_file(audio)
        return output_file


def printed_log(message) -> None:
    logging.info(message)
    print(message)

def print_timelapse(processName:str,old_time:float) -> None:
    printed_log(f"{processName} took {str((time.time() - old_time) * 1000)} ms")

def play_sound(fileName:str) -> None:
    '''
    #FIXME improve writing
    @param fileName: String, path to audio file
    takes audiofile and tries to play it
    '''
    if 'winsound' in mod:
        winsound.PlaySound(fileName, winsound.SND_FILENAME)
    else:
        call(["aplay", fileName])

def filename_parse(input_text:str) -> str:
    '''
    @param input_text: String, input text to be synthesized
    @return: String, denoting path to saved file
    replaces special characters to create easy to use filename adds ".wav" to obtained filename
    '''
    replace_char_with = { 
        ' ' : '-',
        '.' : '_',
        '!' : '',
        '?' : '',
        '°c' : 'degrees celcius',
        ',' : ''}
    for key in replace_char_with:
        input_text = input_text.replace(key,replace_char_with[key])
    return input_text+".wav"

# ---

# remove file after it was sent 
def remove_audio_file(file_path:str) -> bool:
    try:
        os.remove(file_path)
        printed_log(f"Removed audio file {file_path}")
        return True
    except:
        printed_log(f"Error removing audio file {file_path}")
        return False
    finally:
        return False


# temporarily save audio file to disk
# returns path to saved file
def save_audio_file(audio:numpy.typing.NDArray) -> str:
    ''' 
    @param audio: numpy array, audio data
    @return: String, denoting path to saved audio file
    creates random prefix for filename and saves audio to it, returns the path to it too
    '''
    random_suffix:int = random.randint(0, 1000000)
    output_file_name:str = "GLaDOS-tts-tempfile{}.wav".format(random_suffix)
    output_file = (f"{audio_path}{output_file_name}")

    # Write audio file to disk at 22,05 kHz sample rate
    logging.info(f"Saving audio as {output_file}")
    write(output_file, 22050, audio)
    return output_file

def check_audio_folder() -> None:
    if not os.path.exists('audio'):
        os.makedirs('audio')

def check_audio_file(filename) -> bool:
    complete_path = f"{audio_path}{filename}"
    already_exist = os.path.exists(complete_path)
    # Update access time. This will allow for routine cleanups
    if(already_exist): os.utime(complete_path, None)
    return already_exist

def main():
    #FIXME remove, only acts as library
    printed_log("Initializing TTS Engine...")
    glados = Glados()
    glados.load_glados_model()
    if(len(sys.argv)==2):
        printed_log("Using command line argument as text")
        output_file = glados.generate_tts(sys.argv[1])
        play_sound(output_file)
    else:
        while(1):
            printed_log("Using user input as text")
            input_text = input("Input: ")
            output_file = glados.generate_tts(input_text)
            play_sound(output_file)

if __name__ == "__main__":
    main()
