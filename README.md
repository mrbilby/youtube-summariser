### youtube-summariser

The code in this repo will allow you to summarise and manipulate youtube videos in various ways. This is done by leveraging packages that allow youtube video captions to be downloaded in concert with ChatGPT APIs.

NOTE YOU WILL NEED AN OPENAI API KEY UNLESS YOU SWITCH THE CODE TO USING A LOCAL LLM

## Description

main.py - This code is a terminal based tool that allows you to:
- Create summaries of youtube videos
- Create summaries of epubs
- Create audio of text using ChatGPT TTS
- Create pictures from a text at 30s intervals
- Create slideshow videos using an audio file and text
- Take captions in one language from a youtube video, translate to another language, convert to speech and then match audio length to original video

main_KIVY.py - this code allows you to summarise a video using Kivy as the UI. This code can be converted to run on an iPhone. I have it working now on an iPhone 13. Bear in mind Kivy is a bit awkward to get setup and run but it does work and I find this app particularly useful for YouTube videos where you want a summary rather than have to wait around for them to get to the point.
