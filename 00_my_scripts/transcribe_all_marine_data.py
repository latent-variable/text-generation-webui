import whisperx
import time
import os
import json
import numpy as np
from datetime import timedelta
from multiprocessing import Pool



MODEL = True
RELOAD_MODEL = True
LANGUAGES = whisperx.utils.TO_LANGUAGE_CODE
model_a, metadata = None, None
diarize_model = None
params = {
    'whipser_language': 'english',
    'whipser_model': 'large-v3',
    'auto_submit': True,
    'device': "cuda" ,
    'batch_size': '8',
    'compute_type': "float16",
    'reduce_noise': False,
    'hms': False,

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

def file_transcription(audio_file):
    global MODEL, RELOAD_MODEL, LANGUAGES, diarize_model, model_a, metadata

    tik = time.time()

    if audio_file is None:
        return "No valid audio file.", None 
    
    print(audio_file)
    
    if not audio_file.lower().endswith('.mp3'):
        return "Please upload an mp3 file.", None
    
   
    device = params["device"]
    batch_size = int(params["batch_size"])
    compute_type = params["compute_type"]
    whisper_language = params["whipser_language"]
    whisper_model = params["whipser_model"]

    # decode the language
    whisper_language = LANGUAGES[whisper_language]

    #reduce noise
    # if params['reduce_noise']:
    #     audio_file = reduce_noise_on_file(audio_file)
    #     duration = time.time() - tik
    #     print(f'Time to reduce noise: {str(timedelta(seconds=int(duration)))}')

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
    if model_a is None:
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
            if diarize_model is None:
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


    return result["segments"], format_transcript_results(result["segments"], hms=False)

def format_timedelta(t):
    hours, remainder = divmod(t.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Format timedelta as hh:mm:ss.sss
    return "{:02}:{:02}:{:02}.{:03}".format(hours, minutes, seconds, t.microseconds//1000)


def format_transcript_results(transcript_results, hms=False):
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
        if hms:
            start_time = format_timedelta(timedelta(seconds=entry['start']))
            end_time = format_timedelta(timedelta(seconds=entry['end']))
            transcription.append(f"{start_time}{ARROW}{end_time}{ARROW}{speaker}: {entry['text']}")
        else:
            start_time = entry['start']
            end_time = entry['end']
            transcription.append(f"{start_time}{ARROW}{end_time}{ARROW}{speaker}: {entry['text']}")
    
    return "\n".join(transcription)




def get_count_of_files(folder_path):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp3'):
                count += 1
    return count



def process_files(files_data):
    total_file_count = len(files_data)
    file_count = 0
    for file_data in files_data:
        folder_path, root, output_folder_path, file = file_data
        if file.endswith('.mp3'):
            try:
                file_count += 1
                # replace the base folder path with the output folder path
                new_root  = root.replace(folder_path, output_folder_path)
                output_json_file = os.path.join(new_root, file.replace('.mp3', '.json'))

                # skip if ouputfile already exists
                if os.path.exists(output_json_file):
                    print(f'**File {output_json_file} already exists. Skipping...')
                    continue

                audio_file = os.path.join(root, file)
                print(f'\n{file_count} out of {total_file_count} Processing file: {audio_file}')

                transcription_json, transcription_txt = file_transcription(audio_file)

                # ensure the directory exists
                os.makedirs(new_root, exist_ok=True)
                # write to json file
                with open(output_json_file, 'w') as f:
                    json.dump(transcription_json, f)

                # write to txt file
                with open(output_json_file.replace('.json', '.txt'), 'w') as f:
                    f.write(transcription_txt)
            except Exception as e:
                print(f'Error processing file {audio_file}: {e}')
                continue


def traverse_folder(folder_path, output_folder_path, num_buckets):
    files_data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp3'):
                new_root  = root.replace(folder_path, output_folder_path)
                output_json_file = os.path.join(new_root, file.replace('.mp3', '.json'))
                # skip if ouputfile already exists
                if os.path.exists(output_json_file):
                    print(f'**File {output_json_file} already exists. Skipping...')
                    continue
            files_data.append((folder_path, root, output_folder_path, file))

    #shuffle the files
    np.random.shuffle(files_data)

    # split the files into num_buckets buckets
    buckets = np.array_split(files_data, num_buckets)
    print(f'Processing {len(files_data)} files into {len(buckets)} buckets')

    # create num_buckets processes
    with Pool(num_buckets) as p:
        p.map(process_files, buckets)

if __name__ == '__main__':
    input_folder = r'D:\Marine_audio'
    output_folder_path = r'D:\Marine_audio_output'
    traverse_folder(input_folder, output_folder_path, num_buckets=3)
    