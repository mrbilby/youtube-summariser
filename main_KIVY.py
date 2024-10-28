import os
import tempfile
from urllib.parse import urlparse, parse_qs
import re
import requests

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.config import ConfigParser
from kivy.core.window import Window

import threading

# External Libraries
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
# import openai  # Removed since server handles OpenAI interactions

# --------------------- Configuration ---------------------

# Initialize ConfigParser
config = ConfigParser()

# Define the path for the config file
CONFIG_FILE = 'config.ini'

# Read existing config or create a new one
if not os.path.exists(CONFIG_FILE):
    config.adddefaultsection('Server')
    config.write()
else:
    config.read(CONFIG_FILE)

# Server Configuration
SERVER_URL = 'SERVER'  # Replace with your actual server URL
API_KEY = 'API_KEY'  # Replace with your actual API key
# Alternatively, read from config or environment
# API_KEY = config.get('Server', 'api_key', fallback=os.getenv('API_KEY'))

# --------------------- Helper Functions ---------------------

def show_popup(title, message, duration=None):
    """
    Displays a popup with the given title and message.
    If duration is set, the popup will auto-dismiss after the specified time (in seconds).
    """
    content = BoxLayout(orientation='vertical', padding=10, spacing=10)
    label = Label(text=message, text_size=(Window.width * 0.8, None), halign='left', valign='middle')
    label.bind(size=label.setter('text_size'))
    content.add_widget(label)
    
    if duration is None:
        btn = Button(text='Close', size_hint=(1, 0.25))
        content.add_widget(btn)
        popup = Popup(title=title, content=content, size_hint=(0.8, 0.4))
        btn.bind(on_release=popup.dismiss)
    else:
        popup = Popup(title=title, content=content, size_hint=(0.8, 0.4))
        Clock.schedule_once(lambda dt: popup.dismiss(), duration)
    
    popup.open()
    return popup if duration is None else None

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
            return transcript_text, language_code
        except NoTranscriptFound:
            # Iterate through available transcripts and pick the first one
            for transcript in transcript_list:
                transcript_data = transcript.fetch()
                transcript_text = " ".join([entry['text'] for entry in transcript_data])
                language_code = transcript.language_code
                return transcript_text, language_code
            return None, None
    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        return None, None

def request_summary_from_server(captions):
    """
    Sends captions to the server and retrieves the generated summary.
    """
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY  # Include the API key in the headers
    }
    payload = {
        'captions': captions
    }
    
    try:
        response = requests.post(SERVER_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        if 'summary' in data:
            return data['summary']
        else:
            return f"Error: {data.get('error', 'Unknown error.')}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"

def generate_video_summary(captions):
    """
    Sends captions to the server to generate a summary.
    """
    return request_summary_from_server(captions)

# --------------------- Kivy Screens ---------------------

class HomeScreen(Screen):
    """
    The Home Screen with buttons to navigate to Transcript Management and API Key Management.
    """
    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
    
        # Title Label
        label = Label(text="YouTube AI Assistant", font_size=24, size_hint=(1, 0.2))
        layout.add_widget(label)
    
        # "YouTube Summarise" Button
        btn_summarise = Button(text="YouTube Summarise", size_hint=(1, 0.15))
        btn_summarise.bind(on_release=self.go_to_transcript)
        layout.add_widget(btn_summarise)
    
        # "Set API Key" Button (Optional: Remove if not needed)
        # btn_set_api = Button(text="Set API Key", size_hint=(1, 0.15))
        # btn_set_api.bind(on_release=self.go_to_api_key)
        # layout.add_widget(btn_set_api)
    
        self.add_widget(layout)
    
    def go_to_transcript(self, instance):
        self.manager.current = 'transcript'
    
    # def go_to_api_key(self, instance):
    #     self.manager.current = 'apikey'

class TranscriptScreen(Screen):
    """
    The Transcript Management Screen where users can input YouTube URL and get summaries.
    """
    def __init__(self, **kwargs):
        super(TranscriptScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
    
        # Title Label
        label = Label(text="Transcript Management", font_size=20, size_hint=(1, 0.1))
        layout.add_widget(label)
    
        # YouTube URL Input
        self.url_input = TextInput(
            hint_text="Enter YouTube URL",
            size_hint=(1, 0.1),
            multiline=False
        )
        layout.add_widget(self.url_input)
    
        # "Summarise" Button
        btn_summarise = Button(text="Summarise", size_hint=(1, 0.1))
        btn_summarise.bind(on_release=self.process_transcript)
        layout.add_widget(btn_summarise)
    
        # Summary Display Area
        self.summary_label = Label(
            text="",
            size_hint=(1, None),  # Dynamic height
            text_size=(0.95 * Window.width, None),  # 95% of the width for better readability
            halign='left',
            valign='top'
        )
        self.summary_label.bind(texture_size=self.update_label_height)
        self.summary_label.bind(width=self.update_text_size)
    
        # ScrollView for Summary
        scroll = ScrollView(
            size_hint=(1, 0.7),
            do_scroll_x=False,
            do_scroll_y=True
        )
        scroll.add_widget(self.summary_label)
        layout.add_widget(scroll)
    
        # "Back to Home" Button
        btn_back = Button(text="Back to Home", size_hint=(1, 0.1))
        btn_back.bind(on_release=self.go_back)
        layout.add_widget(btn_back)
    
        self.add_widget(layout)
        self.processing_popup = None  # Initialize processing_popup
    
    def on_pre_enter(self):
        """
        No longer need to check for API key.
        """
        pass  # Remove the API key check
    
    def update_text_size(self, instance, value):
        """
        Updates the text_size of the summary_label based on its current width.
        This ensures that text wraps appropriately within the available space.
        """
        self.summary_label.text_size = (self.summary_label.width * 0.95, None)
    
    def update_label_height(self, instance, texture_size):
        """
        Updates the height of the summary_label based on its content.
        """
        self.summary_label.height = texture_size[1]
    
    def go_back(self, instance):
        self.manager.current = 'home'
    
    def process_transcript(self, instance):
        """
        Initiates the transcript processing in a separate thread.
        """
        youtube_url = self.url_input.text.strip()
        if not youtube_url:
            show_popup("Error", "Please enter a YouTube URL.")
            return
    
        video_id = get_video_id(youtube_url)
        if not video_id:
            show_popup("Error", "Invalid YouTube URL.")
            return
    
        # Start processing in a new thread to keep UI responsive
        threading.Thread(target=self.handle_transcript, args=(video_id,)).start()
    
    def handle_transcript(self, video_id):
        """
        Handles fetching transcript and generating summary.
        Updates the UI accordingly.
        """
        # Show a single "Processing" popup on the main thread
        Clock.schedule_once(lambda dt: self.show_processing_popup())
    
        # Fetch transcript
        transcript_text, language_code = check_transcript(video_id)
    
        if transcript_text:
            # Generate summary by sending request to server
            summary = generate_video_summary(transcript_text)
            # Schedule updating the summary and dismissing the popup on the main thread
            Clock.schedule_once(lambda dt: self.update_summary(summary))
        else:
            # Dismiss the processing popup
            Clock.schedule_once(lambda dt: self.dismiss_processing_popup())
            # Show error popup
            Clock.schedule_once(lambda dt: show_popup("Error", "Transcript not available for this video.", duration=3))
    
    def show_processing_popup(self):
        """
        Shows the processing popup and stores the popup instance.
        """
        self.processing_popup = show_popup("Processing", "Fetching transcript and generating summary...", duration=None)
    
    def update_summary(self, summary):
        """
        Updates the summary_label with the generated summary and dismisses the processing popup.
        """
        # Update the summary label
        self.summary_label.text = summary
    
        # Dismiss the processing popup
        self.dismiss_processing_popup()
    
        # Optionally, show a success popup (auto-dismiss after 2 seconds)
        show_popup("Success", "Summary generated successfully.", duration=2)
    
    def dismiss_processing_popup(self):
        """
        Dismisses the processing popup if it's still open.
        """
        if self.processing_popup:
            self.processing_popup.dismiss()
            self.processing_popup = None

# --------------------- Screen Manager ---------------------

class MyScreenManager(ScreenManager):
    """
    Manages screen transitions.
    """
    pass

# --------------------- Main App ---------------------

class YouTubeSummariseApp(App):
    """
    The main Kivy App.
    """
    def build(self):
        # Initialize ScreenManager with Fade Transition
        sm = ScreenManager(transition=FadeTransition())

        # Add Screens
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(TranscriptScreen(name='transcript'))

        return sm

# --------------------- Run App ---------------------

if __name__ == '__main__':
    YouTubeSummariseApp().run()
