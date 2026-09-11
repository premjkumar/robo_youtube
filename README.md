Robo-YouTube Fedora Assistant

An autonomous local robotic companion built for Fedora Linux. Powered by Qwen2.5-0.5B-Instruct running locally via Ollama, this assistant features native LangChain tool-binding, live internet search, ad-blocked YouTube audio streaming, and a full bi-directional voice interface (microphone input and local TTS response).
Features

    Local AI Processing: Runs entirely offline on your hardware via Ollama using the ultra-lightweight Qwen2.5 (0.5B) model.

    Live Internet Search: Fetches real-time web results and news instantly using DuckDuckGo.

    Ad-Blocked YouTube Audio Streaming: Automatically searches YouTube, extracts high-quality audio, and strips out ads and sponsored segments using yt-dlp with SponsorBlock integration.

    Local Audio Playback: Streams and plays media natively using mpv.

    Voice Interface: Listens to microphone input via SpeechRecognition and talks back using local pyttsx3 / espeak-ng text-to-speech.

Prerequisites

Before running the agent, make sure your Fedora system has the required media and system packages installed:
Bash

sudo dnf install mpv ffmpeg espeak-ng portaudio-devel -y

Ensure Ollama is installed and running on your system:
Bash

curl -fsSL https://ollama.com/install.sh | sh
ollama serve

Pull the required Qwen model variant in a separate terminal:
Bash

ollama pull qwen2.5:0.5b-instruct

Installation

    Clone the repository:
    Bash

    git clone https://github.com/premjkumar/robo_youtube.git
    cd robo_youtube

    Install the required Python dependencies:
    Bash

    pip install --user langchain-ollama langchain-core duckduckgo-search yt-dlp SpeechRecognition pyttsx3

Usage

Run the robot assistant script:
Bash

python robot_agent.py

    Voice Mode: Press [Enter] at the prompt to speak into your microphone.

    Text Mode: Type your query directly into the prompt (e.g., "Play some lo-fi beats" or "What is the latest space news?").

    Exit: Type exit, quit, or q to shut down the system.
