import librosa
import noisereduce as nr
from pydub import AudioSegment
import numpy as np

def reduce_noise_on_file(file_path):
    # load data
    data, rate = librosa.load(file_path, sr=None)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=False)
    print('reduced_noise created')

    # Convert the reduced noise from float to int16 to prepare for export
    reduced_noise_int16 = (np.iinfo(np.int16).max * (reduced_noise/np.abs(reduced_noise).max())).astype(np.int16)

    # Create an audio segment from the numpy array and export it as an MP3
    audio_segment = AudioSegment(reduced_noise_int16.tobytes(), frame_rate=rate, sample_width=reduced_noise_int16.dtype.itemsize, channels=1)
    audio_segment.export(file_path.replace('.mp3', '_reduce_noise.mp3'), format="mp3")




if __name__ == '__main__':
    reduce_noise_on_file(r'C:\Users\linoa\Downloads\audio\audio.mp3')