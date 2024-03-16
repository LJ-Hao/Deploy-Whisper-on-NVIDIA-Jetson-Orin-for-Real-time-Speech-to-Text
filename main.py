#! python3.7
import numpy as np
import speech_recognition as sr
import whisper
import re

from datetime import datetime, timedelta
from queue import Queue
from flask import Flask, render_template
import threading
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on('audio')
def audio():
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    # load model
    audio_model = whisper.load_model("base.en")    

    record_timeout = 2
    phrase_timeout = 3

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                result = audio_model.transcribe(audio_np, fp16=False, temperature=0.0)
                text = result['text'].strip()

                if '!!!!!' in text:
                    continue
                if 'a little bit of a little bit of'in text:
                    continue
                if 'I\'m sorry.I\'m sorry.I\'m sorry.' in text:
                    continue
                if 'Okay. Okay. Okay' in text:
                    continue

                # if phrase_complete:
                #     transcription.append(text)
                # else:
                #     transcription[-1] = text
                
                # if  transcription[-1] != "":
                    # # Clear the console to reprint the updated transcription.
                    # os.system('cls' if os.name=='nt' else 'clear')
                    # for line in transcription:
                socketio.emit('transcription', text) 
                    # Flush stdout.
                    # print('', end='', flush=True)
                # socketio.emit('transcription', text+'\n') 

        except KeyboardInterrupt:
            break

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    thread = threading.Thread(target=audio)
    thread.daemon = True
    thread.start()
    socketio.run(app, debug=True)
