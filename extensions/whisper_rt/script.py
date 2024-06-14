import time
import threading
import numpy as np
import gradio as gr
import noisereduce as nr
import soundfile as sf
import webrtcvad
from collections import deque
from pydub import AudioSegment
import whisperx
from concurrent.futures import ThreadPoolExecutor


# Create a ThreadPoolExecutor with 16 threads to transcribe the audio in parallel
executor = ThreadPoolExecutor(max_workers=16)

running_avg_buffer = deque(maxlen=30)  # Adjust maxlen to your preference
average_threshold_ratio = 0.5  # Adjust based on experimentation

# Create a VAD object
vad = webrtcvad.Vad()

# Set its aggressiveness mode, which is an integer between 0 and 3. 
# 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
vad.set_mode(3)

# Create a state object to store the audio stream and the transcribed text
audio_stream = np.array([])
text_stream = [""]
audio_lock = threading.Lock()
text_lock = threading.Lock()

# min and max duration of audio
min_duration = 3
max_duration = 10

# call counter
counter = 0

# Create a transcriber
transcriber_large = whisperx.load_model('large-v3', device='cuda', language='en')
transcriber_small= whisperx.load_model("base.en", device='cuda', language='en')


def calculate_n_fft(time_duration_ms, sample_rate):
    time_duration_s = time_duration_ms / 1000.0  # Convert milliseconds to seconds
    n_fft = time_duration_s * sample_rate
    n_fft_power_of_two = 2**np.round(np.log2(n_fft))  # Round to nearest power of two
    return int(n_fft_power_of_two)

def reduce_noise_on_file(y, sr):
    # perform noise reduction
    time_duration_ms = 12  # milliseconds   
    n_fft = calculate_n_fft(time_duration_ms, sr)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=y, sr=sr, stationary=False, thresh_n_mult_nonstationary=1.5,
                                    time_constant_s=1.0,  sigmoid_slope_nonstationary=2.0,  n_fft=n_fft, device='cuda')
    return reduced_noise
   

def vad_filter(y, sr):

    # Reduce noise in the audio data to improve VAD performance
    ry = reduce_noise_on_file(y, sr)

     # Convert the audio data to the correct format
    audio = AudioSegment(ry.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    audio = audio.set_frame_rate(16000)

    # save audio to file
    audio.export('vad.wav', format='wav')

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


def call_large_transcriber(sr):
    global text_lock, audio_lock, audio_stream, min_duration
    # If the stream is longer than 5 seconds, start a new thread to transcribe the audio
    size_samples = min_duration * sr
    if len(audio_stream) > size_samples:
        # Write the stream to a file as an mp3
        with audio_lock:
            stream = np.array(audio_stream)
            audio_stream = np.array([])

        with text_lock:
            # Reset the stream
            text = text_stream[-1]
            index = len(text_stream)-1
            text_stream.append("")
           
        if text != "":
            transcribe_large_chunk(sr, stream, index)
            

def transcribe_large_chunk(sr, audio_stream, index):
    global text_lock, text_stream
    print('large')
    tik = time.time()

    # write audio to file
    sf.write('large.wav', audio_stream, sr)
    segments = transcriber_large.transcribe('large.wav')['segments']
    results = [segment['text'] for segment in segments]
    tok = time.time()
    
     # Acquire the lock before updating the state
    print(f"**Transcribed large in {tok-tik:.3f} seconds, {index}")
    with text_lock:
        text_stream[index] = ' '.join(results)
        
        
def transcribe_small_chunk(audio_stream, sr):
    global counter, text_lock, text_stream
    counter += 1
    if counter % 3 == 0:
        print('small')
        tik = time.time()
        
        # write audio to file
        sf.write('small.wav', audio_stream, sr)
        segments = transcriber_small.transcribe('small.wav')['segments']
        results = [segment['text'] for segment in segments]
        result = ' '.join(results)
        tok = time.time()
        print(f"Transcribed small in {tok-tik:.3f} seconds")
        with text_lock:
            text_stream[-1] = f'<{result}>'
            

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

    # print(f"Current Level: {current_level}, Running Avg: {running_avg}, Threshold: {threshold}")

    return current_level < threshold


def format_audio_stream(y):
    # Convert the audio to the correct format
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return y


def transcribe(sr, y):
    global audio_stream, audio_lock, max_duration
    
    # If the audio is silent, start
    if np.sum(np.abs(y)) == 0:
        call_large_transcriber(sr)
        return 
    
    if is_silence(y):
        call_large_transcriber(sr)
        return
    

    with audio_lock:
        audio_stream= np.concatenate([audio_stream, format_audio_stream(y)])
        stream = np.array(audio_stream)

    # Filter the audio using VAD
    if not vad_filter(y, sr):
        call_large_transcriber(sr)
        return 

    # If the stream is longer than max_duration seconds, start a new thread to transcribe the audio
    if (len(stream) / sr) > max_duration:
        call_large_transcriber(sr)
    else:
        transcribe_small_chunk(stream, sr)


def transcribe_and_update(new_chunk):
    global text_stream, text_lock

    sr, y = new_chunk
    
    # Update the State object
    # transcribe(sr, y)
    # Start a new thread to transcribe the audio
    executor.submit(transcribe, sr, y)


    text = ' '.join(text_stream)
    # Return the transcription to update the Textbox immediately
    return text

def clear_text():
    global text_stream, text_lock
    with text_lock:
        text_stream = [""]
    return ""


def ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                audio = gr.Audio(streaming=True, label="Speak Now")

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                output = gr.Textbox(lines=20, label="Transcription", autoscroll=True, )
                clear_button = gr.Button(value="Clear Text")

                    
        audio.change(transcribe_and_update, inputs=audio, outputs=output, concurrency_limit=1, show_progress=False)
        clear_button.click(clear_text, outputs=output)

    return demo
    

if __name__ == "__main__":
    demo = ui()
    demo.launch(server_port=7888, inbrowser=True )