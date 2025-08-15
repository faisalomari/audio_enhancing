# youtube_mp3_downloader_threads.py
# -*- coding: utf-8 -*-
"""
Download audio from YouTube as MP3 (multi-threaded)
Requires: pip install yt-dlp
"""

import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def download_youtube_audio(video_url, output_filename="output.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_filename,
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }
        ],
        'quiet': False,
        'noplaylist': True
    }

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

urls = [
    "https://www.youtube.com/watch?v=gNswCif6GwU",
    "https://www.youtube.com/watch?v=i5jkrMhRow0",
    "https://www.youtube.com/watch?v=WBKkr2AwJxk",
    "https://www.youtube.com/watch?v=BASbQgPNkvc",
    "https://www.youtube.com/watch?v=5HqZ0hlWWsM",
    "https://www.youtube.com/watch?v=KWecnNPLK_Y",
    "https://www.youtube.com/watch?v=H37Ubp9RnGw",
    "https://www.youtube.com/watch?v=oS8IXC4SKO0"
]

def main():
    with ThreadPoolExecutor(max_workers=4) as executor:  # adjust workers if needed
        futures = []
        for idx, url in enumerate(urls, start=1):
            output_name = f"lectures/{idx}.mp3"
            futures.append(executor.submit(download_youtube_audio, url, output_name))

        for future in as_completed(futures):
            try:
                future.result()
                print("✅ One download completed!")
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
