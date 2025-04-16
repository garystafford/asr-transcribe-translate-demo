"""
Extracts audio from video files using FFmpeg.
Author: Gary A. Stafford
Date: 2025-04-15
"""

import logging
import os
import ffmpeg


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_audio_from_videos(video_directory: str, audio_directory: str) -> int:
    """
    Extracts audio from video files in the specified directory and saves them as MP3 files.
    Args:
        None
    Returns:
        int: The number of videos processed.
    """

    video_count = 0
    for video_file in os.listdir(video_directory):
        video_path = os.path.join(video_directory, video_file)
        if not video_path.lower().endswith((".mp4")):
            logging.warning(f"Skipping {video_file} - not a valid video file")
            continue
        logging.info(f"Extracting audio from {video_file}...")
        video_count += 1
        audio_file = (
            f"{audio_directory}\\{video_path.split('\\')[-1].replace('.mp4', '.mp3')}"
        )
        ffmpeg.input(video_path).output(
            audio_file, ac=1, ar=16_000, loglevel="quiet"
        ).overwrite_output().run()

    return video_count
