


import os, random, json
import numpy as np
from scipy import signal
from pydub import AudioSegment
from pydub.generators import Triangle
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

with open("genre_symbols_mapping.json", "r") as file:
    genre_id_to_name = json.load(file)


def make_click_video(v_idx, data, fps=60):
    """Create synchronized click videos (with and without original audio)
    for a randomly selected sample from a dataset entry.
    
    Args:
        data (dict): Dataset containing motion/audio metadata.
        fps (float): Sampling rate of anchor sequences.
    """
    # ----- Select random index -----
    ky = 'bothhand_y_bothfoot_y_torso_y_uni'
    hit_idx = data[ky]["hit_idx"]
    # v_idx = random.randint(1, len(hit_idx) - 1)

    # ----- Extract sample metadata -----
    hit_bpm = data[ky]["hit_bpm"]
    hit_beat_pulse = data[ky]["hit_beat_pulse"]
    hit_ref_bpm = data[ky]["hit_ref_bpm"]
    hit_seg_names = data[ky]["hit_seg_names"]
    hit_anc_seq = data[ky]["hit_anc_seq"]
    hit_genres = data[ky]["hit_genres"]
    filename = data[ky]["filename"]

    fname = filename[v_idx].replace("cAll", "c01")
    anchor_sequence = hit_anc_seq[v_idx].flatten()
    gen = genre_id_to_name.get(str(hit_genres[v_idx]), "Unknown")
    hit_seg = hit_seg_names[v_idx]
    beat_pulse = hit_beat_pulse[v_idx]

    # ----- Load video -----
    video_path = os.path.join("./aist_dataset/video", fname + ".mp4")
    video = VideoFileClip(video_path)

    # ----- Compute onset times -----
    time = np.arange(len(anchor_sequence)) / fps
    peaks, _ = signal.find_peaks(beat_pulse[:len(anchor_sequence)],
                                 height=0.1, distance=fps/6)
    onset_times = time[peaks]

    # ----- Generate click audio -----
    click_duration = 25  # ms
    click_freq = 1200  # Hz
    click = Triangle(click_freq).to_audio_segment(duration=click_duration)
    total_duration = (len(anchor_sequence) / fps) * 1000

    audio_click = AudioSegment.silent(duration=total_duration)
    for onset in onset_times:
        position = int(onset * 1000)
        audio_click = audio_click.overlay(click, position=position)

    click_dir = "./aist_dataset/video_examples"
    os.makedirs(click_dir, exist_ok=True)
    click_path = os.path.join(click_dir, "wavs", f"{v_idx}_clicktrack.wav")
    audio_click.export(click_path, format="wav")

    # ----- Merge with video -----
    click_audio = AudioFileClip(click_path)

    # (1) Video + click only
    video.without_audio().with_audio(click_audio).write_videofile(
        os.path.join(click_dir, "with_clicks", f"{v_idx}_{gen}_{hit_seg}_video_click.mp4"),
        codec="libx264", audio_codec="aac", fps=video.fps
    )

    # (2) Video + original audio + click
    combined_audio = CompositeAudioClip([video.audio, click_audio])
    video.with_audio(combined_audio).write_videofile(
        os.path.join(click_dir, "with_audio" , f"{v_idx}_{gen}_{hit_seg}_video_audio_click.mp4"),
        codec="libx264", audio_codec="aac", fps=video.fps
    )

    print(f"✅ Exported click videos for {fname} ({gen}, {hit_seg})")
