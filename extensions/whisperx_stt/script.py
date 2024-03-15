import os
import wave
import time
import whisperx
import gradio as gr
import threading
from datetime import timedelta
import speech_recognition as sr

from modules import shared

LANGUAGES = whisperx.utils.TO_LANGUAGE_CODE


input_hijack = {
    'state': False,
    'value': ["", ""]
}


writers = ["txt", "vtt", "srt", "tsv", "json", "aud"]
       

# default whisper parameters
params = {
    'whipser_language': 'english',
    'whipser_model': 'large-v3',
    'auto_submit': True,
    'device': "cuda" ,
    'batch_size': '16',
    'compute_type': "float16",
    'reduce_noise': False,
    'hms': False,
    'just_text':False,

    'diarize': True,
    'min_speakers': 1,
    'max_speakers': 20,


    'vad_onset': 0.5,
    'vad_offset': 0.363,
    'chunk_size': 30,
    'temperature': 0,
    'best_of': 5,
    'beam_size': 5,
    'patience': 1.0,
    'length_penalty': 1.0,
    'suppress_tokens': "-1",
    'suppress_numerals': False,
    'initial_prompt': None,
    'condition_on_previous_text': False,
    'fp16': True,
    'temperature_increment_on_fallback': 0.2,
    'compression_ratio_threshold': 2.4,
    'logprob_threshold': -1.0,
    'no_speech_threshold': 0.6
}

# just text preset 
just_text_params = params.copy()
just_text_params['just_text'] = True
just_text_params['diarize'] = False


# noisy preset
noisy_params = params.copy()
noisy_params['reduce_noise'] = True
noisy_params['compression_ratio_threshold'] = 3.0
noisy_params['logprob_threshold'] = -2.0
noisy_params['diarize'] = False 



# Preset the parameters
presets = {
    'default': params.copy(),
    'just_text': just_text_params,
    'noisy': noisy_params
}

# add a lock to prevent multiple threads from accessing the same resource simultaneously
lock = threading.Lock()

# add a global variable to store the model
MODEL = None
RELOAD_MODEL = True
    
def chat_input_modifier(text, visible_text, state):
    global input_hijack
    if input_hijack['state']:
        input_hijack['state'] = False
        return input_hijack['value']
    else:
        return text, visible_text

def save_temp_file(audio_data, filename="temp.wav"):

    # Open a new WAV file
    with wave.open(filename, 'wb') as f:
        # Set the parameters
        f.setnchannels(1)  # mono audio
        f.setsampwidth(audio_data.sample_width)
        f.setframerate(audio_data.sample_rate)

        # Write the audio data
        f.writeframes(audio_data.frame_data)

def delete_temp_file(filename="temp.wav"):
    if os.path.exists(filename):
        os.remove(filename)

def do_stt(audio, whisper_model, whisper_language):
    global MODEL, RELOAD_MODEL, LANGUAGES
    transcription = ""
    r = sr.Recognizer()
    with lock:
        device = params["device"]
        batch_size = int(params["batch_size"])
        compute_type = params["compute_type"]

        # decode the language
        whisper_language = LANGUAGES[whisper_language]

        print(f'do_stt- whisper_language: {whisper_language}, whisper_model: {whisper_model}, device: {device}, batch_size: {batch_size}, compute_type: {compute_type}')

        # Convert to AudioData
        audio_data = sr.AudioData(sample_rate=audio[0], frame_data=audio[1], sample_width=4)
        # Save the audio data to a file
        save_temp_file(audio_data)
        
        # Load the audio
        audio = whisperx.load_audio("temp.wav")

        if MODEL is None or RELOAD_MODEL:
            RELOAD_MODEL = False
            
            # Load the parameters
            v_params = get_vad_params()
            w_params = get_whisper_params()

            # Load the model
            MODEL = whisperx.load_model(whisper_model, device, compute_type=compute_type, asr_options=w_params, vad_options=v_params)


        # Transcribe the audio
        result = MODEL.transcribe(audio, language=whisper_language, batch_size=batch_size, chunk_size=params["chunk_size"]) 

        # Load the alignment model
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

        # Align the transcription
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        transcription = get_text_from_segment(result["segments"])
        
        # delete the temp file
        delete_temp_file()

        return transcription

def get_text_from_segment(segments):

    text = ""
    for segment in segments:
        text += segment['text'] + " "
    return text


def auto_transcribe(audio, auto_submit, whipser_model, whipser_language):
    if audio is None:
        return "", ""
    transcription = do_stt(audio, whipser_model, whipser_language)
    if auto_submit:
        input_hijack.update({"state": True, "value": [transcription, transcription]})

    return transcription, None

def format_timedelta(t):
    hours, remainder = divmod(t.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Format timedelta as hh:mm:ss.sss
    return "{:02}:{:02}:{:02}.{:03}".format(hours, minutes, seconds, t.microseconds//1000)


def format_transcript_results(transcript_results, hms=False, just_text=False):
    transcription = []

    ARROW = "	"  
    speaker = ''
    for entry in transcript_results:
        
        if 'words' in entry:
            words = entry['words']
           
            for word in words:
                if "speaker" in word:
                    speaker = word['speaker']
                    break
        if just_text:
            transcription.append(entry['text'])
        elif hms:
            start_time = format_timedelta(timedelta(seconds=entry['start']))
            end_time = format_timedelta(timedelta(seconds=entry['end']))
            transcription.append(f"{start_time}{ARROW}{end_time}{ARROW}{speaker}: {entry['text']}")
        else:
            start_time = entry['start']
            end_time = entry['end']
            transcription.append(f"{start_time}{ARROW}{end_time}{ARROW}{speaker}: {entry['text']}")

    return "\n".join(transcription)

def get_vad_params():
    '''
    Get the VAD parameters from the settings
    '''
    return {
        "vad_onset": float(params['vad_onset']),
        "vad_offset": float(params['vad_offset'])
    }

def get_whisper_params():
    '''
    Get the whisper parameters from the settings
    '''
    return {
        "beam_size": int(params['beam_size']),
        "patience": float(params['patience']),
        "length_penalty": float(params['length_penalty']),
        "temperatures": float(params['temperature']),
        "compression_ratio_threshold": float(params['compression_ratio_threshold']),
        "log_prob_threshold": float(params['logprob_threshold']),
        "no_speech_threshold": float(params['no_speech_threshold']),
        "condition_on_previous_text": params['condition_on_previous_text'],
        "initial_prompt": params['initial_prompt'],
        "suppress_tokens": [int(x) for x in params['suppress_tokens'].split(",")],
        "suppress_numerals": params['suppress_numerals'],
    }

def reduce_noise_on_file(file_path):
    try:
        import librosa
        import noisereduce as nr
        from pydub import AudioSegment
        import numpy as np

        print('Reducing noise on file')
        # load data
        data, rate = librosa.load(file_path, sr=None)
        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=False)
        print('Reduced noise on file completed')

        # Convert the reduced noise from float to int16 to prepare for export
        reduced_noise_int16 = (np.iinfo(np.int16).max * (reduced_noise/np.abs(reduced_noise).max())).astype(np.int16)

        # Create an audio segment from the numpy array and export it as an MP3
        audio_segment = AudioSegment(reduced_noise_int16.tobytes(), frame_rate=rate, sample_width=reduced_noise_int16.dtype.itemsize, channels=1)
        file_path = file_path.replace('.mp3', '_reduce_noise.mp3')
        audio_segment.export(file_path, format="mp3")
        return file_path
    except Exception as e:
        print(f'Exception: {e}')
        return file_path


def file_transcription(audio_file):
    global MODEL, RELOAD_MODEL, LANGUAGES

    tik = time.time()

    if audio_file is None:
        return "No valid audio file.", None 
    
    print(audio_file)
    
    if not audio_file.lower().endswith('.mp3'):
        return "Please upload an mp3 file.", None
    
    with lock:
        device = params["device"]
        batch_size = int(params["batch_size"])
        compute_type = params["compute_type"]
        whisper_language = params["whipser_language"]
        whisper_model = params["whipser_model"]

        # decode the language
        whisper_language = LANGUAGES[whisper_language]

        #reduce noise
        if params['reduce_noise']:
            audio_file = reduce_noise_on_file(audio_file)
        
        duration = time.time() - tik
        print(f'Time to reduce noise: {str(timedelta(seconds=int(duration)))}')

        print(f'file_transcription- whipser_language: {whisper_language}, whipser_model: {whisper_model}')
        print(f'audio_file: {audio_file}, device: {device}, batch_size: {batch_size}, compute_type: {compute_type}')
    
        # Load the model
        if MODEL is None or RELOAD_MODEL:
            RELOAD_MODEL = False

            v_params = get_vad_params()
            w_params = get_whisper_params()

            # Load the model
            MODEL = whisperx.load_model(whisper_model, device, language=whisper_language, compute_type=compute_type, asr_options=w_params, vad_options=v_params)

        # Load the audio
        audio = whisperx.load_audio(audio_file)
        duration = time.time() - tik
        print(f'Time to load audio: {str(timedelta(seconds=int(duration)))}')

        # Transcribe the audio
        result = MODEL.transcribe(audio, language=whisper_language, batch_size=batch_size, chunk_size=params["chunk_size"]) 
        duration = time.time() - tik
        print(f'Time to transcribe: {str(timedelta(seconds=int(duration)))}')
        
        # Load the alignment model
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        duration = time.time() - tik
        print(f'Time to load aligment model: {str(timedelta(seconds=int(duration)))}')

        # Align the transcription
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        duration = time.time() - tik
        print(f'Time to align: {str(timedelta(seconds=int(duration)))}')

        # get authentification token from file
        auth_token = None
        if os.path.exists("auth_token.txt"):
            with open("auth_token.txt", "r") as f:
                auth_token = f.read().strip()   
        
        if params['diarize']:
        
            try:
                # Load the diarization model
                diarize_model = whisperx.DiarizationPipeline( use_auth_token=auth_token, device=device)
                duration = time.time() - tik
                print(f'Time to load diarization model: {str(timedelta(seconds=int(duration)))}')

                # Diarize the segments
                diarize_segments = diarize_model(audio, min_speakers=params['min_speakers'], max_speakers=params['max_speakers'])
                duration = time.time() - tik
                print(f'Time to diarize: {str(timedelta(seconds=int(duration)))}')

                # Assign speaker labels
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                print(f'Exception: {e}')
                print(f'Remeber to add your auth token to auth_token.txt, or ensure that you have the correct permissions to access the diarization model.')
        
        duration = time.time() - tik
        print(f"Transcription completed in {str(timedelta(seconds=int(duration)))}")

        return format_transcript_results(result["segments"], hms=params['hms'], just_text=params['just_text']), None


# Add a change handler for the dropdown
def update_settings(preset_name):
    global RELOAD_MODEL, presets
    if preset_name not in presets:
        print(f"Warning: preset '{preset_name}' not found.")
        return
    
    # Reset the model to be reloaded so parameters are updated
    if RELOAD_MODEL is not None:
        RELOAD_MODEL = None
    
    settings = presets[preset_name]
    

    return  settings['whipser_language'], settings['whipser_model'], settings['auto_submit'], \
            settings['device'], settings['batch_size'], settings['compute_type'], \
            settings['reduce_noise'], settings['hms'], settings['just_text'], \
            settings['diarize'], settings['min_speakers'], settings['max_speakers'], \
            settings['vad_onset'], settings['vad_offset'], settings['chunk_size'], \
            settings['temperature'], settings['best_of'], settings['beam_size'], \
            settings['patience'], settings['length_penalty'], settings['suppress_tokens'], \
            settings['suppress_numerals'], settings['initial_prompt'], \
            settings['condition_on_previous_text'], settings['fp16'], \
            settings['temperature_increment_on_fallback'], \
            settings['compression_ratio_threshold'], \
            settings['logprob_threshold'], settings['no_speech_threshold']


def reload_model(value):
    global MODEL, RELOAD_MODEL
    RELOAD_MODEL = True
    MODEL = None
    print("Model will reloaded on next transcription.")

def ui():
    global RELOAD_MODEL, params, presets
    with gr.Accordion("Whisper Voice Recording ", open=False):
        with gr.Row():
            audio = gr.Audio(source="microphone")
            auto_submit = gr.Checkbox(label='Submit the transcribed audio automatically', value=params['auto_submit'])

    with gr.Accordion("Whisper Settings", open=False):
        
        with gr.Row():
            # Create a dropdown for the presets
            preset_dropdown = gr.Dropdown(choices=list(presets.keys()), label='Presets', value='default')

        with gr.Accordion("Basic Settings", open=False):
            device = gr.Dropdown(label = "Device", value= params["device"], choices=["cuda", "cpu"])
            batch_size = gr.Dropdown(label = "Batch Size", value= params["batch_size"], choices=["1", "2", "4", "8", "16", "32", "64", "128", "256"])
            compute_type = gr.Dropdown(label = "Compute Type", value= params["compute_type"], choices=["int8", "float16", "float32"])
            whipser_model = gr.Dropdown(label='Whisper Model', value=params['whipser_model'], choices=[ "small.en", "medium","large-v2", "large-v3"])
            whipser_language = gr.Dropdown(label='Whisper Language', value=params['whipser_language'], choices=list(LANGUAGES.keys()) )
            reduce_noise = gr.Checkbox(label='Reduce noise ', value=params['reduce_noise'] )
            hms = gr.Checkbox(label='Show time in hours, minutes, and seconds', value=params['hms'])
            just_text = gr.Checkbox(label='Only return the text', value=params['just_text'])
            

        with gr.Accordion("VAD Settings", open=False):
            vad_onset = gr.Textbox(label='VAD Onset', value=str(params['vad_onset']))
            vad_offset = gr.Textbox(label='VAD Offset', value=str(params['vad_offset']))
            chunk_size = gr.Textbox(label='Chunk Size', value=str(params['chunk_size']))

        with gr.Accordion("Diarization Settings", open=False):
            diarize = gr.Checkbox(label='Apply Diarization', value=params['diarize'])
            min_speakers = gr.Textbox(label='Minimum Speakers', value=str(params['min_speakers']))
            max_speakers = gr.Textbox(label='Maximum Speakers', value=str(params['max_speakers']))

        with gr.Accordion("Whisper Advance settings", open=False):
            temperature = gr.Textbox(label='Temperature', value=str(params['temperature']))
            best_of = gr.Textbox(label='Best of', value=str(params['best_of']))
            beam_size = gr.Textbox(label='Beam Size', value=str(params['beam_size']))
            patience = gr.Textbox(label='Patience', value=str(params['patience']))
            length_penalty = gr.Textbox(label='Length Penalty', value=str(params['length_penalty']))
            suppress_tokens = gr.Textbox(label='Suppress Tokens', value=str(params['suppress_tokens']))
            suppress_numerals = gr.Checkbox(label='Suppress Numerals', value=params['suppress_numerals'])
            initial_prompt = gr.Textbox(label='Initial Prompt', value=str(params['initial_prompt']), placeholder="Enter problematic words to assist the model in generating the transcription.")
            condition_on_previous_text = gr.Checkbox(label='Condition on Previous Text', value=params['condition_on_previous_text'])
            fp16 = gr.Checkbox(label='FP16', value=params['fp16'])
            temperature_increment_on_fallback = gr.Textbox(label='Temperature Increment on Fallback', value=str(params['temperature_increment_on_fallback']))
            compression_ratio_threshold = gr.Textbox(label='Compression Ratio Threshold', value=str(params['compression_ratio_threshold']))
            logprob_threshold = gr.Textbox(label='Logprob Threshold', value=str(params['logprob_threshold']))
            no_speech_threshold = gr.Textbox(label='No Speech Threshold', value=str(params['no_speech_threshold']))
        
        with gr.Row():
            reload_model_button = gr.Button(label="Reload Model", type="button", value="Reload Model")


    with gr.Row():
        with gr.Column(align="center"):
            audio_file = gr.Audio(label="or Upload Audio File", type='filepath', source="upload", format='mp3')
            transcribe_button = gr.Button(label="Generate", type="button", value="Transcribe")
            output = gr.Textbox(lines=20, label="Transcription")
           

    
    audio.change(
        auto_transcribe, [audio, auto_submit, whipser_model, whipser_language], [shared.gradio['textbox'], audio]).then(
        None, auto_submit, None, _js="(check) => {if (check) { document.getElementById('Generate').click() }}")
    
    # Preset dropdown
    preset_dropdown.change(update_settings, preset_dropdown, outputs=[ whipser_language, whipser_model, auto_submit,
                                                                       device, batch_size, compute_type, reduce_noise, 
                                                                       hms, just_text, diarize, min_speakers, max_speakers,
                                                                       vad_onset, vad_offset, chunk_size, temperature,
                                                                       best_of, beam_size, patience, length_penalty, 
                                                                       suppress_tokens, suppress_numerals, initial_prompt, 
                                                                       condition_on_previous_text, fp16, 
                                                                       temperature_increment_on_fallback, 
                                                                       compression_ratio_threshold, 
                                                                       logprob_threshold, no_speech_threshold])

    # Basic settings 
    whipser_model.change(lambda x: params.update({"whipser_model": x}), whipser_model, None)
    whipser_language.change(lambda x: params.update({"whipser_language": x}), whipser_language, None)
    auto_submit.change(lambda x: params.update({"auto_submit": x}), auto_submit, None)
    device.change(lambda x: params.update({"device": x}), device, None)
    batch_size.change(lambda x: params.update({"batch_size": x}), batch_size, None)
    compute_type.change(lambda x: params.update({"compute_type": x}), compute_type, None)
    reduce_noise.change(lambda x: params.update({"reduce_noise": x}), reduce_noise, None)
    hms.change(lambda x: params.update({"hms": x}), hms, None)
    just_text.change(lambda x: params.update({"just_text": x}), just_text, None)


    # Diarization settings
    diarize.change(lambda x: params.update({"diarize": x}), diarize, None)
    min_speakers.change(lambda x: params.update({"min_speakers": int(x)}), min_speakers, None)
    max_speakers.change(lambda x: params.update({"max_speakers": int(x)}), max_speakers, None)

    # VAD settings
    vad_onset.change(lambda x: params.update({"vad_onset": float(x)}), vad_onset, None)
    vad_offset.change(lambda x: params.update({"vad_offset": float(x)}), vad_offset, None)
    chunk_size.change(lambda x: params.update({"chunk_size": float(x)}), chunk_size, None)

    # Whisper Advance settings
    temperature.change(lambda x: params.update({"temperature": float(x)}), temperature, None)
    best_of.change(lambda x: params.update({"best_of": int(x)}), best_of, None)
    beam_size.change(lambda x: params.update({"beam_size": int(x)}), beam_size, None)
    patience.change(lambda x: params.update({"patience": float(x)}), patience, None)
    length_penalty.change(lambda x: params.update({"length_penalty": float(x)}), length_penalty, None)
    suppress_tokens.change(lambda x: params.update({"suppress_tokens": x}), suppress_tokens, None)
    suppress_numerals.change(lambda x: params.update({"suppress_numerals": x}), suppress_numerals, None)
    initial_prompt.change(lambda x: params.update({"initial_prompt": x}), initial_prompt, None)
    condition_on_previous_text.change(lambda x: params.update({"condition_on_previous_text": x}), condition_on_previous_text, None)
    fp16.change(lambda x: params.update({"fp16": x}), fp16, None)
    temperature_increment_on_fallback.change(lambda x: params.update({"temperature_increment_on_fallback": float(x)}), temperature_increment_on_fallback, None)
    compression_ratio_threshold.change(lambda x: params.update({"compression_ratio_threshold": float(x)}), compression_ratio_threshold, None)
    logprob_threshold.change(lambda x: params.update({"logprob_threshold": float(x)}), logprob_threshold, None)
    no_speech_threshold.change(lambda x: params.update({"no_speech_threshold": float(x)}), no_speech_threshold, None)

    reload_model_button.click(reload_model, reload_model_button, None)

    # Add the new components to the interface
    transcribe_button.click(fn=file_transcription, inputs=audio_file, outputs=[output, audio_file])
