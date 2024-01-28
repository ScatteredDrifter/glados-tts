# --- / 
# -- / File contains logic for processing text to speech
# -- | using torch and hifigan


# --- / 
# -- / external imports
from typing import final
import torch
import random
import logging
from scipy.io.wavfile import write
import time
import os
from sys import modules as mod
import sys

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
    glados_model = None
    vocoder = None
    device = None


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
        self.glados_model = torch.jit.load('models/glados.pt')
        self.vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=self.device)

        for i in range(4):
            init = self.glados_model.generate_jit(prepare_text(str(i)))
            init_mel = init['mel_post'].to(self.device)
            init_vo = self.vocoder(init_mel)
        printed_log(f"Models loaded.")

    def load_glados_model(self):
        option_devices = ["vulkan","cuda"]
        while(self.glados_model==None):
            self.device = self.get_available_device(option_devices)
            try:
                self.load_models()
            except:
                printed_log(f"Exception loading device "+self.device)
                if(self.device != 'cpu'): option_devices.remove(self.device)
                else: sys.exit()

    def get_audio_from_text(self,text):
    	# Tokenize, clean and phonemize input text
        phonemized_text = prepare_text(text).to(self.device)
        with torch.no_grad():
            # Generate generic TTS-output
            old_time = time.time()
            tts_output = self.glados_model.generate_jit(phonemized_text)

            # Use HiFiGAN as vocoder to make output sound like GLaDOS
            mel = tts_output['mel_post'].to(self.device)
            audio = self.vocoder(mel)
            print_timelapse("The audio sample: ",old_time)

            # Normalize audio to fit in wav-file
            audio = audio.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16')
        return audio

    # Generate audio file from given text as string
    def generate_tts(self, input_text:str):
        filename = filename_parse(input_text)
        audio_file_exist = check_audio_file(filename)
        if (audio_file_exist):
            printed_log("The audio sample sent from cache.")
            output_file = f"{audio_path}{filename}"
        else:
            audio = self.get_audio_from_text(input_text)
            output_file = save_audio_file(audio,filename)
        return output_file

def printed_log(message):
    logging.info(message)
    print(message)

def print_timelapse(processName,old_time):
    printed_log(f"{processName} took {str((time.time() - old_time) * 1000)} ms")

def play_sound(fileName):
    if 'winsound' in mod:
        winsound.PlaySound(fileName, winsound.SND_FILENAME)
    else:
        call(["aplay", fileName])

def filename_parse(input_text):
    replace_char_with = { 
        ' ' : '-',
        '.' : '_',
        '!' : '',
        '?' : '',
        '°c' : 'degrees celcius',
        ',' : ''}
    for key in replace_char_with:
        input_text = input_text.replace(key,replace_char_with[key])
    # filename = input_text.replace(" ", "-")
    # filename = filename.replace(".", "_")
    # filename = filename.replace("!", "")
    # filename = filename.replace("?", "")
    # filename = filename.replace("°c", "degrees celcius")
    # filename = filename.replace(",", "")+".wav"
    return input_text+".wav"

# remove file after it was sent 
def remove_audio_file(file_path:str) -> bool:
    try:
        os.remove(file_path)
        return True
    except:
        printed_log(f"Error removing audio file {file_path}")
        return False
    finally:
        return False

# temporarily save audio file to disk
def save_audio_file(audio,filename=None):
    random_suffix:int = random.randint(0, 1000000)
    output_file_name:str = "GLaDOS-tts-tempfile{}.wav".format(random_suffix)
    # if(filename and len(filename)<200):
        # output_file_name = filename
    output_file = (f"{audio_path}{output_file_name}")
    # Write audio file to disk at 22,05 kHz sample rate
    logging.info(f"Saving audio as {output_file}")
    write(output_file, 22050, audio)
    return output_file

def check_audio_folder():
    if not os.path.exists('audio'):
        os.makedirs('audio')

def check_audio_file(filename):
    complete_path = f"{audio_path}{filename}"
    already_exist = os.path.exists(complete_path)
    # Update access time. This will allow for routine cleanups
    if(already_exist): os.utime(complete_path, None)
    return already_exist

def main():
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
