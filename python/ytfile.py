#!/usr/bin/env python
# File: ytfile.py
# Name: D.Saravanan
# Date: 03/03/2023

""" Script to download a file from YouTube with pytube """

from pytube import YouTube

link = "https://www.youtube.com/watch?v=XV46ALr3OMg"
file = YouTube(link)

print("************************* Video Title ****************************")
print(file.title)

print("************************* Tumbnail Image *************************")
print(file.thumbnail_url)

# set stream resolution
resolution = file.streams.get_highest_resolution()

# get stream resolution
for stream in file.streams:
    print(stream)

print("************* filtering by progressive Streams ********************")
""" progressive streams have the video and audio in a single file, 
    but typically do not provide the highest quality media """
for pstream in file.streams.filter(progressive=True):
	print(pstream)

print("************* filtering by adaptive Steams ************************")
""" adaptive streams split the video and audio tracks
    but can provide much higher quality """
for astream in file.streams.filter(adaptive=True):
	print(astream)

print("************* filtering for audio-only streams ********************")
""" query the streams that contain only the audio track """
for audiostream in file.streams.filter(only_audio=True):
	print(audiostream)

print("**************** filtering for MP4 streams ************************")
""" query only streams in the MP4 format """
for mp4stream in file.streams.filter(file_extension='mp4'):
	print(mp4stream)

print("************************* Download Video *************************")
#file.download()
