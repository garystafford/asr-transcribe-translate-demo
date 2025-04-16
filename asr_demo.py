"""
Video Transcription using ASR (Automatic Speech Recognition) to transcribe audio from video files.
Author: Gary A. Stafford
Date: 2025-04-14
"""

# Standard library imports
import logging
import os
import time
import json
import shutil
import sys

# Third-party library imports
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset

# Local application/library imports
import utilities.audio_extractor as audio_extractor
import utilities.translator as Translator

# Constants
VIDEO_DIRECTORY = "videos"
AUDIO_DIRECTORY = "audio"
OUTPUT_DIRECTORY = "output"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Main function to set up the environment, extract audio from videos, transcribe audio, translate text, and save results.
    Args:
        None
    Returns:
        None
    """

    # Choose the model to use for transcription from command line (0-5)
    model_choice = (int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    model = get_model_list()[model_choice]
    model_id = model["model_id"]
    output_file = os.path.join(
        OUTPUT_DIRECTORY, f"transcriptions_{model['model_name']}.json"
    )

    # Set up the environment
    logging.info("Setting up the environment...")
    setup_environment(output_file)

    # Extract audio from videos in the specified directory
    logging.info("Extracting audio from videos...")
    start = time.time()
    video_count = audio_extractor.extract_audio_from_videos(
        VIDEO_DIRECTORY, AUDIO_DIRECTORY
    )
    if video_count == 0:
        logging.error("No valid video files found in the directory.")
        return
    end = time.time()
    total_extract_audio_time_sec = round(end - start, 2)
    logging.info(f"Extracted audio from {video_count} videos.")
    logging.info(f"Total audio extraction time: {total_extract_audio_time_sec} seconds")

    # Load the audio dataset
    logging.info("Loading audio dataset...")
    # https://huggingface.co/docs/datasets/audio_load#audiofolder
    audio_dataset = load_dataset("audiofolder", data_dir=AUDIO_DIRECTORY, split="train")

    # Transcribe the audio dataset
    logging.info("Transcribing audio dataset...")
    start = time.time()
    results = generate_transcriptions(model_id, audio_dataset)
    end = time.time()
    total_transcription_time_sec = round(end - start, 2)
    logging.info(f"Total transcription time: {total_transcription_time_sec} seconds")

    # Translate the transcriptions
    logging.info(f"Translating transcriptions...")
    start = time.time()
    translate = Translator.Translator()
    for result in results:
        translation_german = translate.translate_text(
            result["text"], language="deu_Latn"
        )
        result["translation_german"] = translation_german

        translation_chinese = translate.translate_text(
            result["text"], language="zho_Hans"
        )
        result["translation_chinese"] = translation_chinese
    end = time.time()
    total_translation_time_sec = round(end - start, 2)
    logging.info(f"Total translation time: {total_translation_time_sec} seconds")

    # Save the results
    logging.info("Saving results...")
    all_results = {}
    all_results["results"] = results
    all_results["stats"] = {
        "model": model_id,
        "total_video": video_count,
        "total_extract_audio_time_sec": total_extract_audio_time_sec,
        "total_transcription_time_sec": total_transcription_time_sec,
        "total_translation_time_sec": total_translation_time_sec,
        "average_total_time_per_video_sec": round(
            (
                total_extract_audio_time_sec
                + total_transcription_time_sec
                + total_translation_time_sec
            )
            / video_count,
            2,
        ),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    logging.info(f"Results saved to {output_file}")
    logging.info("Done!")


def setup_environment(output_file):
    """
    Sets up the environment by creating necessary directories and deleting existing ones.
    Args:
        None
    Returns:
        None
    """

    if not os.path.exists(VIDEO_DIRECTORY):
        logging.error(f"Video directory {VIDEO_DIRECTORY} does not exist")
        return

    if os.path.exists(AUDIO_DIRECTORY):
        logging.info(f"Deleting existing audio directory {AUDIO_DIRECTORY}...")
        shutil.rmtree(AUDIO_DIRECTORY)
        logging.info(f"Creating audio directory {AUDIO_DIRECTORY}...")
        os.makedirs(AUDIO_DIRECTORY)

    if not os.path.exists(OUTPUT_DIRECTORY):
        logging.info(f"Creating output directory {os.path.dirname(output_file)}...")
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def get_model_list():
    """
    Returns a list of available models for transcription.
    Returns:
        List[Dict]: A list of dictionaries containing model IDs and names.
    """

    models = [
        {
            "model_id": "openai/whisper-tiny",
            "model_name": "openai-whisper-tiny",
        },
        {
            "model_id": "openai/whisper-small",
            "model_name": "openai-whisper-small",
        },
        {
            "model_id": "openai/whisper-medium",
            "model_name": "openai-whisper-medium",
        },
        {
            "model_id": "openai/whisper-large-v3-turbo",
            "model_name": "openai-whisper-large-v3-turbo",
        },
        {
            "model_id": "openai/whisper-large-v3",
            "model_name": "openai-whisper-large-v3",
        },
        {
            "model_id": "distil-whisper/distil-large-v3.5",
            "model_name": "distil-whisper-distil-large-v3.5",
        },
    ]

    return models


def generate_transcriptions(model_id, audio_dataset):
    """
    Transcribes the audio from a dataset of audio files using a pre-trained model.
    Args:
        audio_dataset (Dataset): A dataset containing audio files to be transcribed.
    Returns:
        List[Dict]: A list of transcription results for each audio file.
    """

    # Load the model and processor
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=attn_implementation,
    )

    # Optional model generation_config (here or in generate_kwargs below)
    # model.generation_config.forced_decoder_ids = None
    # model.generation_config.input_ids = model.generation_config.forced_decoder_ids
    # model.generation_config.language = "hindi"
    # model.generation_config.task = "transcribe"  # options: translate or transcribe
    logging.debug(model.generation_config)

    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Create the pipeline
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,  # batch size for inference - set based on your device
        torch_dtype=model.dtype,
        device=model.device,
        return_timestamps=True,
        generate_kwargs={
            # "forced_decoder_ids": None,
            # "language": "english",
            # "max_new_tokens": 128
            "task": "transcribe",
        },
    )

    results = []

    # Process the audio files in the dataset
    for idx, out in enumerate(
        asr_pipeline(
            KeyDataset(audio_dataset, "audio"),
        )
    ):
        audio_file = audio_dataset[idx]["audio"]["path"].split("\\")[-1]
        logging.info(f"Transcribing/translating {audio_file}...")
        out["audio_file"] = audio_file
        logging.debug(audio_dataset[idx])
        logging.debug(json.dumps(out, indent=4))
        logging.info(f"Transcription/translation result: {out['text']}")
        results.append(out)
    return results


if __name__ == "__main__":
    main()
