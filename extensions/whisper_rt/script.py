import time
import threading
import numpy as np
import gradio as gr
import soundfile as sf
import webrtcvad
from collections import deque
from pydub import AudioSegment
from transformers import pipeline

running_avg_buffer = deque(maxlen=30)  # Adjust maxlen to your preference
average_threshold_ratio = 0.5  # Adjust based on experimentation

# Create a VAD object
vad = webrtcvad.Vad()

# Set its aggressiveness mode, which is an integer between 0 and 3. 
# 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
vad.set_mode(3)

# Create a state object to store the audio stream and the transcribed text
state = {'stream': np.array([]), 'text': [""]}
lock = threading.Lock()

# min and max duration of audio
min_duration = 2
max_duration = 10

#
counter = 0

# Create a transcriber
transcriber_large = pipeline("automatic-speech-recognition", model="openai/whisper-medium.en", device=0)
transcriber_tiny = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=0)

def vad_filter(y, sr):
     # Convert the audio data to the correct format
    audio = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    audio = audio.set_frame_rate(16000)

    # Split the audio into 10 ms frames
    frame_duration_ms = 10  # Duration of a frame in ms
    bytes_per_sample = 2
    frame_byte_count = int(sr * frame_duration_ms / 1000) * bytes_per_sample  # Number of bytes in a frame
    frames = [audio.raw_data[i:i+frame_byte_count] for i in range(0, len(audio.raw_data), frame_byte_count)]

    voice_activity = False
    # Use VAD to check if each frame contains speech
    for frame in frames:
        if len(frame) != frame_byte_count:
            continue  # Skip frames that are not exactly 10 ms long
        
        contains_speech = vad.is_speech(frame, sample_rate=16000)
        
        if contains_speech:
            voice_activity = True
            break
    return voice_activity

def transcribe_using_large_model(sr, stream, index):
    global lock, state
    print("Transcribing...",index)
    # write audio to file
    tik = time.time()
    result = transcriber_large({"sampling_rate": sr, "raw": stream, "language": "en"})["text"]
    tok = time.time()
    print(f"**Transcribed large in {tok-tik:.3f} seconds, {index}")
    # print(result)

    # Acquire the lock before updating the state
    with lock:
        state['text'][index] = result

def call_transcribe_thread(sr):
    global lock, state, min_duration
    # If the stream is longer than 5 seconds, start a new thread to transcribe the audio
    size_samples = min_duration * sr
    if len(state['stream']) > size_samples:
        # Write the stream to a file as an mp3
        with lock:
            # Reset the stream
            stream = state['stream']
            state['stream'] = np.array([])
            index = len(state['text'])-1
            state['text'].append("")
            
        transcribe_thread = threading.Thread(target=transcribe_using_large_model, args=(sr,stream, index))
        transcribe_thread.start()

def transcribe_using_small_model(audio_stream, sr):
    global counter
    counter += 1
    if counter % 2 == 0:
        tik = time.time()
        result = transcriber_tiny({"sampling_rate": sr, "raw": audio_stream, "language": "en"})["text"]
        tok = time.time()
        print(f"Transcribed small in {tok-tik:.3f} seconds")

        with lock:
            state['text'][-1] = f'<{result}>'

def update_running_average(new_value):
    running_avg_buffer.append(new_value)
    return np.mean(running_avg_buffer)


def is_silence(audio_data, base_threshold=150):
    """Check if the given audio data represents silence based on running average."""
    current_level = np.abs(audio_data).mean()
    running_avg = update_running_average(current_level)

    # if running_avg < base_threshold: # this might indicate just silence 
    #     return False
    
    dynamic_threshold = running_avg * average_threshold_ratio
    threshold = max(dynamic_threshold, base_threshold)

    print(f"Current Level: {current_level}, Running Avg: {running_avg}, Threshold: {threshold}")

    return current_level < threshold
def transcribe(sr, y):
    global state, lock, max_duration

    # If the audio is silent, start
    if np.sum(np.abs(y)) == 0:
        # print("No audio")
        call_transcribe_thread(sr)
        return 
    
    if is_silence(y):
        print("Silence detected")
        call_transcribe_thread(sr)
        return
    
    # Filter the audio using VAD
    if not vad_filter(y, sr):
        # print("No voice activity detected")
        call_transcribe_thread(sr)
        return 
    
    # Convert the audio to the correct format
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    
    # Concatenate the new audio to the stream
    with lock:
        state['stream'] = np.concatenate([state['stream'], y])

    len_stream = len(state['stream']) /sr
    # print(f'stream length{len_stream} seconds') 
    # If the stream is longer than 5 seconds, start a new thread to transcribe the audio
    if len(state['stream']) /sr > max_duration:
        call_transcribe_thread(sr)
    else:
        transcribe_using_small_model(state['stream'], sr)


    return 

def transcribe_and_update(new_chunk):
    global state, lock

    sr, y = new_chunk
    
    # Update the State object
    transcribe_thread = threading.Thread(target=transcribe, args=(sr, y))
    transcribe_thread.start()
  
    # Return the transcription to update the Textbox immediately
    with lock:
        return_string = ' '.join(state['text'])
        
    return return_string


def ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                audio = gr.Audio(source='microphone', streaming=True, label="Speak Now")

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                output = gr.Textbox(lines=20, label="Transcription", autoscroll=True)

                    
        audio.change(transcribe_and_update, inputs=audio, outputs=output)

    return demo
    

if __name__ == "__main__":
    demo = ui()
    demo.launch()