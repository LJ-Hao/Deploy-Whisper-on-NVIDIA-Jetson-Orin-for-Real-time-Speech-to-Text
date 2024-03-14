#! python3.7
import numpy as np
import speech_recognition as sr
import whisper

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
    recorder.energy_threshold = 100
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    # load model
    audio_model = whisper.load_model("base")    

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
            now = datetime.now()
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
                # # If we detected a pause between recordings, add a new item to our transcription.
                # # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                
                if  transcription[-1] != "":
                    # # Clear the console to reprint the updated transcription.
                    # os.system('cls' if os.name=='nt' else 'clear')
                    # for line in transcription:
                    socketio.emit('transcription', transcription[-1]+'\n') 
                    # Flush stdout.
                    # print('', end='', flush=True)
                else:
                    pass
                # Infinite loops are bad for processors, must sleep.
                # sleep(0.25)
        except KeyboardInterrupt:
            break

    # print("\n\nTranscription:")
    # for line in transcription:
    #     print(line)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    thread = threading.Thread(target=audio)
    thread.daemon = True
    thread.start()
    socketio.run(app)
