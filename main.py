import os
import sys
import tempfile
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import math
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv
import re
from bs4 import BeautifulSoup
from ebooklib import epub
import ebooklib
from ebooklib import ITEM_DOCUMENT
import tiktoken
import math
from mutagen.mp3 import MP3
import textwrap
import requests
import subprocess
import tempfile

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
AudioSegment.converter = "ffmpeg"

FINAL_AUDIO_FILENAME = "transcript_audio.mp3"
CHUNK_SIZE = 4000  # Max characters per chunk to stay below 4096 limit

def generate_video_summary(captions):
    client = OpenAI(api_key=openai_api_key)
    print("Generating summary")
    print("calling GPT")
    try:
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""
                 You create comprehensive and accurate summaries from captions of a video, 
                 explaining concepts and keeping the overall tone of the video."""},
                {"role": "user", "content": f"""The captions of the video are: {captions}. Please summarise."""}
            ]
        )
        generated_text = response.choices[0].message.content.strip()
    except Exception as e:
        generated_text = f"Error: {str(e)}"
    return generated_text

def question_video_summary(summary, question):
    client = OpenAI(api_key=openai_api_key)
    print("Generating summary")
    print("calling GPT")
    try:
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""
                 You answer questions on video summaries from a user and give comprehensive answers drawing on your wider knowledge where needed."""},
                {"role": "user", "content": f"""Please answer this {question} based on this summary {summary}."""}
            ]
        )
        generated_text = response.choices[0].message.content.strip()
    except Exception as e:
        generated_text = f"Error: {str(e)}"
    return generated_text

def get_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    Supports various YouTube URL formats.
    """
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    else:
        return None

def check_transcript(video_id):
    """
    Checks if a transcript exists for the given video ID.
    Returns the transcript text and language code if available, else (None, None).
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try to fetch English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
            transcript_data = transcript.fetch()
            transcript_text = " ".join([entry['text'] for entry in transcript_data])
            language_code = transcript.language_code
            print("English transcript found.")
            return transcript_text, language_code
        except NoTranscriptFound:
            print("English transcript not found. Checking for other language transcripts.")
            # Iterate through available transcripts and pick the first one
            for transcript in transcript_list:
                transcript_data = transcript.fetch()
                transcript_text = " ".join([entry['text'] for entry in transcript_data])
                language_code = transcript.language_code
                print(f"Transcript found in language: {language_code}")
                return transcript_text, language_code
            # If no transcripts found
            print("No transcript found in any language.")
            return None, None
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
        return None, None
    except NoTranscriptFound:
        print("No transcript found for this video.")
        return None, None
    except Exception as e:
        print(f"An error occurred while fetching transcript: {e}")
        return None, None

def translate_text(client, text, source_lang, target_lang='en'):
    """
    Translates text from source_lang to target_lang using OpenAI's ChatCompletion.
    """
    try:
        # Split the text into manageable chunks for translation if necessary
        chunks = split_text_into_chunks(text, 4000)  # Keeping buffer below 4096
        translated_chunks = []
        for idx, chunk in enumerate(chunks, 1):
            prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{chunk}"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use "gpt-4" if available
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            generated_text = response.choices[0].message.content.strip()
            print(f"Translated chunk {idx}/{len(chunks)}.")
            translated_chunks.append(generated_text)
        
        full_translated_text = " ".join(translated_chunks)
        return full_translated_text
    except Exception as e:
        print(f"Error translating text: {e}")
        return None

def split_text_into_chunks(text, max_chars):
    """
    Splits text into chunks not exceeding max_chars, preferably at sentence boundaries.
    """
    import re
    sentences = re.split('(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def get_video_duration(youtube_url):
    """
    Retrieves the duration of the YouTube video in seconds without downloading it.
    """
    try:
        yt = YouTube(youtube_url)
        video_duration = yt.length  # Duration in seconds
        print(f"Video duration: {video_duration} seconds")
        return video_duration
    except Exception as e:
        print(f"Error retrieving video duration: {e}")
        return None

def generate_speech_mp3(client, transcript_text, output_path):
    """
    Uses OpenAI's Text-to-Speech API to generate an MP3 from the transcript.
    Splits the text into chunks if necessary to comply with input limits.
    """
    voiceChoice = input("Which voice?").strip()
    try:
        # Split the transcript into chunks to adhere to the 4096 character limit
        chunks = split_text_into_chunks(transcript_text, 4000)  # Keeping buffer below 4096
        speech_segments = []
        
        for idx, chunk in enumerate(chunks, 1):
            print(f"Generating speech for chunk {idx}/{len(chunks)}...")
            response = client.audio.speech.create(
                model="tts-1",      # Ensure this model is correct as per the latest documentation
                voice=voiceChoice,      # Ensure this voice is supported
                input=chunk
            )

            # Handle the response correctly without using 'with_streaming_response()'
            # Directly access the binary content and write to a temporary file
            temp_audio_path = os.path.join(os.path.dirname(output_path), f"temp_chunk_{idx}.mp3")
            with open(temp_audio_path, 'wb') as f:
                f.write(response.content)
            speech_segments.append(temp_audio_path)
            print(f"Chunk {idx} speech generated at {temp_audio_path}")
        
        # Concatenate all speech segments into the final audio file
        combined = AudioSegment.empty()
        for segment in speech_segments:
            audio = AudioSegment.from_file(segment)
            combined += audio
            os.remove(segment)  # Remove temporary chunk file
        
        combined.export(output_path, format="mp3")
        print(f"Combined speech MP3 generated at {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating speech MP3: {e}")
        return None


def adjust_audio_speed(audio_path, target_duration, output_path):
    """
    Adjusts the speed of the audio to match the target duration using FFmpeg.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        current_duration = len(audio) / 1000.0  # in seconds
        speed_change = current_duration / target_duration

        print(f"Adjusting audio speed by a factor of {speed_change:.4f} to match video duration.")

        # FFmpeg's atempo filter accepts values between 0.5 and 2.0.
        # For values outside this range, chain multiple atempo filters.
        if speed_change < 0.5 or speed_change > 2.0:
            tempos = []
            temp_speed = speed_change
            while temp_speed < 0.5 or temp_speed > 2.0:
                if temp_speed > 2.0:
                    tempos.append(2.0)
                    temp_speed /= 2.0
                elif temp_speed < 0.5:
                    tempos.append(0.5)
                    temp_speed *= 2.0
            tempos.append(temp_speed)
            atempo_filters = ",".join([f"atempo={tempo}" for tempo in tempos])
        else:
            atempo_filters = f"atempo={speed_change}"

        # Use FFmpeg to adjust speed
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name

        command = [
            "ffmpeg",
            "-y",  # Overwrite if exists
            "-i", audio_path,
            "-filter:a", atempo_filters,
            temp_audio_path
        ]
        subprocess_result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if subprocess_result.returncode != 0:
            print(f"FFmpeg error: {subprocess_result.stderr.decode()}")
            return None

        # Load adjusted audio and export
        adjusted_audio = AudioSegment.from_file(temp_audio_path)
        adjusted_audio.export(output_path, format="mp3")
        os.remove(temp_audio_path)
        print(f"Adjusted audio saved at {output_path}")
        return output_path
    except Exception as e:
        print(f"Error adjusting audio speed: {e}")
        return None

def integrate_synopses(previous, new):
    # Remove any redundant headings or introductory phrases from the new content
    new = re.sub(r'^(Plot Summary|Themes|Continued Synopsis).*?\n', '', new, flags=re.IGNORECASE|re.MULTILINE)
    
    # Trim new content to start from the first new full sentence after the last sentence of the previous synopsis
    last_sentence_end = previous.rfind('.')
    first_sentence_end = new.find('.', new.find('.') + 1)
    if first_sentence_end != -1:
        new = new[first_sentence_end + 1:]

    # Combine the synopses with a smooth transition, trimming spaces
    combined = f"{previous.strip()} {new.strip()}"

    return combined

def synopsis_build(epub_file):
    try:
        book = epub_file
        content = ""
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                content += soup.get_text() + "\n\n"
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(content)
        max_tokens = 100000
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

        client = OpenAI()
        full_synopsis = ""
        for i, chunk in enumerate(chunks):
            chunk_text = encoding.decode(chunk)
            prompt = f"Create or update a detailed synopsis by integrating the following text from the book..."

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert literary analyst. Provide a seamless and coherent synopsis, avoiding repetition and ensuring all content is well integrated. Do not write it in the first person."},
                    {"role": "user", "content": f"""Previous synopsis and new text:
                    {full_synopsis}

                    New text to summarize:
                    {chunk_text}
                    """}
                ],
                temperature=0.7
            )
            new_content = response.choices[0].message.content.strip()
            full_synopsis = integrate_synopses(full_synopsis, new_content)

        # Final Review Step
        final_prompt = "Review the entire synopsis for coherence, remove any repetition, and improve the overall flow. Make the synopsis engaging and concise:"
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a literary editor tasked with refining a complete book synopsis."},
                {"role": "user", "content": final_prompt + full_synopsis}
            ],
            temperature=0.7
        )
        final_synopsis = final_response.choices[0].message.content.strip()

    
        return final_synopsis
    except Exception as e:
        print(f"Error with synopsis generation: {e}")


def get_mp3_length(mp3_path):
    """
    Returns the length of the MP3 file in seconds.
    """
    audio = MP3(mp3_path)
    return int(audio.info.length)


def download_image(image_url, save_path):
    """
    Downloads an image from the specified URL and saves it to the given path.
    
    Args:
        image_url (str): The URL of the image to download.
        save_path (str): The local file path where the image will be saved.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred.
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Image successfully downloaded and saved to {save_path}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while downloading the image: {http_err}")
    except Exception as err:
        print(f"An error occurred while downloading the image: {err}")

def split_text_into_parts(text, num_images, max_length=900):
    """
    Splits the text into exactly num_images parts, each not exceeding max_length characters.
    Splits at sentence boundaries to maintain context. If a sentence exceeds max_length,
    it is truncated and appended with an ellipsis.

    Args:
        text (str): The text to split.
        num_images (int): The desired number of parts to split the text into.
        max_length (int, optional): Maximum number of characters per part. Defaults to 900.

    Returns:
        List[str]: A list of text parts.
    """
    import math
    import re

    # Split text into sentences using regex to handle various punctuation.
    # This regex splits on '.', '!' or '?' followed by a space or end of string.
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text.strip())

    # Clean sentences and ensure they end with proper punctuation
    sentences = [s.strip() if s.endswith(('.', '!', '?')) else s.strip() + '.' for s in sentences if s.strip()]

    total_sentences = len(sentences)

    # If number of images exceeds number of sentences, adjust num_images
    if num_images > total_sentences:
        print(f"Warning: Requested {num_images} images but only {total_sentences} sentences available.")
        num_images = total_sentences if total_sentences > 0 else 1

    # Calculate the base number of sentences per part and the remainder
    base_sentences = total_sentences // num_images
    remainder = total_sentences % num_images

    parts = []
    current_index = 0

    for i in range(num_images):
        # Determine the number of sentences for this part
        num_sentences_in_part = base_sentences + (1 if i < remainder else 0)
        part_sentences = sentences[current_index:current_index + num_sentences_in_part]
        current_index += num_sentences_in_part

        # Join sentences to form the part
        part = ' '.join(part_sentences)

        # If part exceeds max_length, truncate appropriately
        if len(part) > max_length:
            # Truncate to max_length - 3 and add ellipsis
            truncated_part = part[:max_length - 3].rsplit(' ', 1)[0] + "..."
            parts.append(truncated_part)
            print(f"Truncated part {i + 1} to fit the max_length.")
        else:
            parts.append(part)

    # In case there are still parts left (due to truncation), fill them with a period
    while len(parts) < num_images:
        parts.append(".")

    return parts



def generate_image(prompt, index, output_dir, size="1024x1024", style="in a consistent art style squished horizontally as image will be stretched to wide screen"):
    """
    Generates an image using OpenAI's Image API based on the prompt.
    Downloads and saves the image to the specified output directory with a consistent naming convention.
    
    Args:
        prompt (str): The text prompt to generate the image.
        index (int): The index of the image (used for naming).
        output_dir (str): The directory where images will be saved.
        size (str, optional): The size of the generated image. Defaults to "1024x1024".
        style (str, optional): Additional style instructions for the image. Defaults to "in a consistent art style".
    """
    try:
        # Combine the prompt with the desired style
        full_prompt = f"{prompt}, {style}"
        if len(full_prompt) > 1000:
            # Truncate the prompt to fit within the 1000-character limit
            full_prompt = full_prompt[:997] + "..."
            print(f"Prompt truncated to fit the 1000 character limit for image {index + 1}.")

        client = OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=full_prompt,  # Use full_prompt directly
            n=1,
            size=size
        )
        
        # Extract the image URL from the response
        image_url = response.data[0].url
        print(f"Generated image URL: {image_url}")
        
        # Define the path where the image will be saved
        image_path = os.path.join(output_dir, f'image_{index + 1}.png')
        
        # Download and save the image
        download_image(image_url, image_path)
        
    except Exception as e:
        print(f"Failed to generate or download image {index + 1}: {e}")


def generate_image_prompt(synopsis_part):
    """
    Generates a concise, visually descriptive prompt suitable for image generation
    based on a part of the synopsis.
    """
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use "gpt-4" if available
            messages=[
                {"role": "system", "content": """
                You are an assistant that converts text into vivid, concise image prompts suitable for image generation. Focus on visual elements, settings, characters, and actions. Ensure the prompt is less than 1000 characters and complies with OpenAI's content policies. In particular DallE image policy must be complied with so keep the prompt as safe as possible. Remove all violence and sexual content.
                """},
                                {"role": "user", "content": f"""
                Based on the following text, create a descriptive image prompt for an image generator that complies with DallE and openAI policy.:

                {synopsis_part}
                """}
            ],
            temperature=0.7,
            max_tokens=150  # Adjust as needed to keep the prompt concise
        )
        prompt = response.choices[0].message.content.strip()
        # Ensure the prompt is less than 1000 characters
        if len(prompt) > 1000:
            prompt = prompt[:997] + "..."
        return prompt
    except Exception as e:
        print(f"Error generating image prompt: {e}")
        return None

def create_video_from_images_and_audio(image_dir, audio_path, output_video_path, display_durations):
    """
    Creates a video from images and an audio file.

    Args:
        image_dir (str): Directory containing the images.
        audio_path (str): Path to the audio file.
        output_video_path (str): Path where the output video will be saved.
        display_durations (list): List of durations for each image in seconds.
    """
    try:
        # Get sorted list of image files
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
        if len(image_files) == 0:
            print("No image files found in the directory.")
            return

        if len(image_files) != len(display_durations):
            print("Number of images and number of durations do not match.")
            return

        # Create a temporary text file listing the images and durations
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_list_file:
            for img_file, duration in zip(image_files, display_durations):
                img_path = os.path.join(image_dir, img_file)
                temp_list_file.write(f"file '{img_path}'\n")
                temp_list_file.write(f"duration {duration}\n")
            # FFmpeg requires the last image to be specified again without a duration
            temp_list_file.write(f"file '{os.path.join(image_dir, image_files[-1])}'\n")
            temp_file_name = temp_list_file.name

        # Create the slideshow video using FFmpeg
        slideshow_video = os.path.join(tempfile.gettempdir(), 'slideshow.mp4')
        command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_file_name,
            '-vsync', 'vfr',
            '-pix_fmt', 'yuv420p',
            slideshow_video
        ]
        subprocess.run(command, check=True)

        # Combine the slideshow video with the audio
        command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-i', slideshow_video,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_video_path
        ]
        subprocess.run(command, check=True)

        # Clean up temporary files
        os.remove(temp_file_name)
        os.remove(slideshow_video)

        print(f"Video successfully created at {output_video_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error during video creation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_video_with_zoom_effect(image_dir, audio_path, output_video_path, display_durations):
    """
    Creates a video from images and an audio file with a slow zoom-in effect on each image.
    """
    try:
        # Get sorted list of image files
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
        if len(image_files) == 0:
            print("No image files found in the directory.")
            return

        if len(image_files) != len(display_durations):
            print("Number of images and number of durations do not match.")
            return

        video_segments = []

        for idx, (img_file, duration) in enumerate(zip(image_files, display_durations)):
            img_path = os.path.join(image_dir, img_file)
            temp_video = os.path.join(tempfile.gettempdir(), f'temp_video_{idx}.mp4')
            zoom_duration = duration

            # Calculate the number of frames based on duration and frame rate (e.g., 30 fps)
            fps = 30
            total_frames = int(zoom_duration * fps)

            # Define initial and final zoom levels
            initial_zoom = 1.0
            final_zoom = 1.0  # Adjust as desired

            # Use frame-based zoom calculation
            zoompan_filter = (
                f"zoompan="
                f"z='{initial_zoom}+({final_zoom - initial_zoom})*on/{total_frames}':"
                f"d=1:"
                f"x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':"
                f"s=1024x1024"
            )

            command = [
                'ffmpeg',
                '-y',  # Overwrite output files without asking
                '-loop', '1',
                '-i', img_path,
                '-vf', zoompan_filter,
                '-c:v', 'libx264',
                '-t', str(zoom_duration),
                '-pix_fmt', 'yuv420p',
                '-r', str(fps),
                temp_video
            ]

            subprocess.run(command, check=True)
            video_segments.append(temp_video)
            print(f"Processed image {idx + 1}/{len(image_files)} with zoom effect.")

        # Concatenate all video segments
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_concat_file:
            for temp_video in video_segments:
                temp_concat_file.write(f"file '{temp_video}'\n")
            temp_concat_file_name = temp_concat_file.name

        concatenated_video = os.path.join(tempfile.gettempdir(), 'concatenated_video.mp4')
        command = [
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_concat_file_name,
            '-c', 'copy',
            concatenated_video
        ]
        subprocess.run(command, check=True)
        print("All video segments concatenated.")

        # Combine the concatenated video with the audio
        command = [
            'ffmpeg',
            '-y',
            '-i', concatenated_video,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_video_path
        ]
        subprocess.run(command, check=True)
        print(f"Final video created at {output_video_path}")

        # Clean up temporary files
        os.remove(temp_concat_file_name)
        os.remove(concatenated_video)
        for temp_video in video_segments:
            os.remove(temp_video)

    except subprocess.CalledProcessError as e:
        print(f"Error during video creation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def api_call(book_detail):
    print("calling GPT")
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model = 'gpt-4o-mini',
            messages=[
                {"role": "system", "content": f"""
                 You are a novel critic and know everything about books and their impact.."""},
                {"role": "user", "content": f"""Please detail the cultural impact, how this book was different from others, detail about the author and similar books to read. The book is {book_detail}"""}
            ]
        )
        generated_text = response.choices[0].message.content.strip()
    except Exception as e:
        generated_text = f"Error: {str(e)}"
    return generated_text

def main():
    first_decision = input("""1 for transcript management, 2 for synopsis generation, 3 for audio from synopsis, 
                           4 for image generation from synopsis, 5 for video generation: """).strip()
    if first_decision == "1":
        youtube_url = input("Enter the YouTube video URL: ").strip()
        if not youtube_url:
            print("No URL provided. Exiting.")
            sys.exit(1)

        video_id = get_video_id(youtube_url)
        if not video_id:
            print("Invalid YouTube URL. Could not extract video ID. Exiting.")
            sys.exit(1)

        # Initialize the OpenAI client
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            sys.exit(1)

        client = OpenAI(api_key=openai_api_key)

        # Step 1: Check for Transcript
        transcript_text, language_code = check_transcript(video_id)

        if transcript_text:
            print("Transcript is available. Proceeding with processing.")

            choice = input("Generate English audio with 'E' or generate a summary with 'S': ").strip()
            if choice == 'S':
                summary = generate_video_summary(transcript_text)
                print(summary)
                print('\n')
                while True:
                    followup = input("Do you have any followup questions? Y or N?: ").strip()
                    if followup == 'Y':
                        user_question = input("What is your question?: ").strip()
                        output = question_video_summary(summary, user_question)
                        print(output)
                    else:
                        print("Ending program")
                        sys.exit(0)
            elif choice == 'E':
                # Step 2: Translate if necessary
                if language_code != 'en':
                    print(f"Transcript is in '{language_code}'. Translating to English.")
                    translated_text = translate_text(client, transcript_text, language_code, 'en')
                    if not translated_text:
                        print("Failed to translate transcript. Exiting.")
                        sys.exit(1)
                else:
                    translated_text = transcript_text

                # Step 3: Get video duration without downloading
                video_duration = get_video_duration(youtube_url)
                if not video_duration:
                    print("Could not retrieve video duration. Exiting.")
                    sys.exit(1)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    print(f"Using temporary directory: {tmpdirname}")

                    # Step 4: Generate speech MP3 from transcript
                    speech_mp3_path = os.path.join(tmpdirname, "transcript_speech.mp3")
                    generated_speech_path = generate_speech_mp3(client, translated_text, speech_mp3_path)
                    if not generated_speech_path:
                        print("Failed to generate speech MP3. Exiting.")
                        sys.exit(1)

                    # Step 5: Adjust speech speed to match video duration
                    adjusted_speech_path = os.path.join(tmpdirname, FINAL_AUDIO_FILENAME)
                    adjusted_audio = adjust_audio_speed(generated_speech_path, video_duration, adjusted_speech_path)
                    if not adjusted_audio:
                        print("Failed to adjust audio speed. Exiting.")
                        sys.exit(1)

                    # Step 6: Move the final audio to current directory
                    destination_path = os.path.join(os.getcwd(), FINAL_AUDIO_FILENAME)
                    Path(adjusted_speech_path).rename(destination_path)
                    print(f"Final audio saved at {destination_path}")

                    # Optional: Play the final audio
                    # Uncomment the following lines if you wish to play the audio automatically
                    # try:
                    #     if sys.platform == "win32":
                    #         os.startfile(destination_path)
                    #     elif sys.platform == "darwin":
                    #         subprocess.call(["open", destination_path])
                    #     else:
                    #         subprocess.call(["xdg-open", destination_path])
                    # except Exception as e:
                    #     print(f"Error playing audio: {e}")
            else:
                print("No choice made")

        else:
            print("Transcript not available. Exiting.")
    elif first_decision=="2":
        book_path = input("input file name of epub in directory: ").strip()
        book = epub.read_epub(book_path)
        synopsis_out = synopsis_build(book)
        print(synopsis_out)
        book_detail = input("Give detail of the book to create a commentary: ").strip()
        commentary = api_call(book_detail)
        synopsis_out = synopsis_out + "/n " + commentary
        with open("synopsis_out.txt", "w") as file:
            file.write(synopsis_out)

    elif first_decision=="3":
        print("Building audio from synopsis_out.txt file")
        with open("synopsis_out.txt", "r") as file:
            synopsis_out = file.read()
        client = OpenAI()
        generate_speech_mp3(client, synopsis_out, "synopsis_speech.mp3")

    elif first_decision=='4':
        synopsis_path = 'synopsis_out.txt'        # Path to your synopsis text file
        mp3_path = 'synopsis_speech.mp3'      # Path to your MP3 audio file
        output_dir = 'generated_images'      # Directory to save generated images

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read the synopsis text
        with open(synopsis_path, 'r', encoding='utf-8') as file:
            synopsis = file.read().strip()

        # Get MP3 length in seconds
        total_seconds = get_mp3_length(mp3_path)
        print(f"MP3 Length: {total_seconds} seconds")

        # Calculate number of images (one per 30 seconds)
        num_images = math.ceil(total_seconds / 30)
        num_images = max(num_images, 1)  # Ensure at least one image
        print(f"Number of images to generate: {num_images}")

        # Split synopsis into parts
        synopsis_parts = split_text_into_parts(synopsis, num_images, max_length=900)
        print("Synopsis split into parts for image generation.")
        for x in synopsis_parts:
            print(len(x))

        # Generate images for each part
        manual_image = input("Manual image creation? Y/N: ").strip()
        if manual_image == 'Y':
            for idx, part in enumerate(synopsis_parts):
                prompt = generate_image_prompt(part)
                if prompt:
                    print(f"Generating image {idx + 1} with prompt: {prompt}")
                    filename_out = str(idx+1)+'_prompt.txt'
                    with open(filename_out, "w") as file:
                        file.write(prompt)
                else:
                    print(f"Skipping image {idx + 1} due to prompt generation error.")
        elif manual_image == "N":
            for idx, part in enumerate(synopsis_parts):
                prompt = generate_image_prompt(part)
                if prompt:
                    print(f"Generating image {idx + 1} with prompt: {prompt}")
                    generate_image(prompt, idx, output_dir)
                else:
                    print(f"Skipping image {idx + 1} due to prompt generation error.")
        else:
            print("no selection made")
    
    elif first_decision == '5':
        create_video_choice = input("Do you want to create a video with the images and audio? (Y/N): ").strip().upper()

        synopsis_path = 'synopsis_out.txt'        # Path to your synopsis text file
        mp3_path = 'synopsis_speech.mp3'      # Path to your MP3 audio file
        output_dir = '/Users/jamesbailey/Desktop/Python/AI Learning/youTubeAI/generated_images'      # Directory to save generated images

        total_seconds = get_mp3_length(mp3_path)
        print(f"MP3 Length: {total_seconds} seconds")

        # Calculate number of images (one per 30 seconds)
        num_images = math.ceil(total_seconds / 30)
        num_images = max(num_images, 1)  # Ensure at least one image

        with open(synopsis_path, 'r', encoding='utf-8') as file:
            synopsis = file.read().strip()

        synopsis_parts = split_text_into_parts(synopsis, num_images, max_length=900)



        print("Synopsis split into parts for image generation.")
        for x in synopsis_parts:
            print(len(x))
        if create_video_choice == 'Y':
            output_video_path = 'output_video.mp4'  # You can change this to your desired output path

            # Calculate the durations for each image
            total_seconds = get_mp3_length(mp3_path)
            num_images = len(synopsis_parts)
            base_duration = total_seconds // num_images
            remainder = total_seconds % num_images

            display_durations = [base_duration] * num_images
            # Distribute the remainder to the first few images
            for i in range(int(remainder)):
                display_durations[i] += 1

            # Print durations for debugging
            print("Display durations for each image:")
            for idx, duration in enumerate(display_durations):
                print(f"Image {idx + 1}: {duration} seconds")

            create_video_with_zoom_effect(output_dir, mp3_path, output_video_path, display_durations)
        else:
            print("Video creation skipped.")

    else:
        print("no decision made")
    

if __name__ == "__main__":
    main()
