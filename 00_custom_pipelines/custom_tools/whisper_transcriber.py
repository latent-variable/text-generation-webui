"""
title: Audio Transcription using Faster Whisper
author: Lino Valdovinos
funding_url: https://github.com/latent-variable
version: 0.2.0
license: MIT
"""
import os
import json
import time
from faster_whisper import WhisperModel

class Tools:
    def __init__(self, model_size: str = "distil-large-v3", device: str = "cuda", compute_type: str = None):
        """
        Initialize the WhisperModel.

        :param model_size: Size of the model to use: tiny, base, small, medium, large-v1, large-v2, large-v3, distil-large-v3.
        :param device: Device to use for inference: 'cpu', 'cuda', or 'auto'.
        :param compute_type: Type of computation: 'float16', 'int8', 'int8_float16'. If None, it will be set based on the device.
        """
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_audio(self, file_path: str, language: str = "", response_format: str = "json") -> str:
        """
        Transcribe an audio file using the local faster-whisper model.

        :param file_path: The path to the audio file to transcribe.
        :param language: Language of the input audio in ISO-639-1 format (optional).
        :param response_format: Format of the transcript output: json, text, srt, verbose_json, or vtt.

        :return: The transcription result in the specified format.
        """

        if not os.path.isfile(file_path):
            return json.dumps({"error": f"File not found: {file_path}"})

        try:
            # Perform transcription
            segments, info = self.model.transcribe(
                file_path,
                language=language if language else None,
                beam_size=5  # You can adjust this parameter as needed
            )

            # Collect the segments
            segment_list = []
            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                segment_list.append(segment_data)

            # Format the output according to response_format
            output = self._format_output(segment_list, info, response_format)

            return output

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _format_output(self, segments, info, response_format):
        if response_format == "json":
            result = {
                "language": info.language,
                "segments": segments
            }
            output = json.dumps(result, ensure_ascii=False)
        elif response_format == "text":
            output = "".join([segment["text"] for segment in segments])
        elif response_format == "srt":
            # Generate SRT format
            output = ""
            for i, segment in enumerate(segments, start=1):
                start_time = self._format_time_srt(segment["start"])
                end_time = self._format_time_srt(segment["end"])
                text = segment["text"].strip()
                output += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
        elif response_format == "vtt":
            # Generate VTT format
            output = "WEBVTT\n\n"
            for segment in segments:
                start_time = self._format_time_vtt(segment["start"])
                end_time = self._format_time_vtt(segment["end"])
                text = segment["text"].strip()
                output += f"{start_time} --> {end_time}\n{text}\n\n"
        elif response_format == "verbose_json":
            # Output all info
            result = {
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": segments
            }
            output = json.dumps(result, ensure_ascii=False)
        else:
            # Default to json
            result = {
                "language": info.language,
                "segments": segments
            }
            output = json.dumps(result, ensure_ascii=False)
        return output

    def _format_time_srt(self, seconds):
        # SRT format: hours:minutes:seconds,milliseconds
        milliseconds = int((seconds - int(seconds)) * 1000)
        time_str = time.strftime('%H:%M:%S', time.gmtime(seconds))
        return f"{time_str},{milliseconds:03d}"

    def _format_time_vtt(self, seconds):
        # VTT format: hours:minutes:seconds.milliseconds
        milliseconds = int((seconds - int(seconds)) * 1000)
        time_str = time.strftime('%H:%M:%S', time.gmtime(seconds))
        return f"{time_str}.{milliseconds:03d}"
