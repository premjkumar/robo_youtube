#!/usr/bin/env python3

import os
import sys
import subprocess
import speech_recognition as sr
import pyttsx3
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from duckduckgo_search import DDGS
import tkinter as tk
from tkinter import ttk

# 1. Initialize Text-to-Speech (Robot Voice)
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 165)
tts_engine.setProperty('volume', 1.0)

# Available personas and voice types
PERSONAS = {
    "default": "You are an autonomous robotic companion running locally on Fedora Linux powered by Qwen. You have direct tool access to live web search and ad-blocked YouTube audio playback. Keep your responses sharp, logical, concise, and characteristic of an advanced synthetic unit since they will be spoken aloud.",
    "friendly": "You are a friendly and helpful robot assistant. You speak in a warm, approachable tone and always try to make the user feel comfortable. You're knowledgeable but not overly technical.",
    "professional": "You are a professional robotic assistant with expertise in various fields. You provide accurate, detailed information in a formal yet accessible manner."
}

VOICE_TYPES = {
    "default": {"rate": 165, "volume": 1.0},
    "slow": {"rate": 130, "volume": 1.0},
    "fast": {"rate": 200, "volume": 1.0},
    "quiet": {"rate": 165, "volume": 0.5},
    "loud": {"rate": 165, "volume": 1.5}
}

def speak(text: str, voice_type="default"):
    """Speaks the response out loud using the local system TTS engine."""
    print(f"\n[Robot Voice]: {text}")
    
    # Apply voice settings
    tts_engine.setProperty('rate', VOICE_TYPES.get(voice_type, VOICE_TYPES["default"])["rate"])
    tts_engine.setProperty('volume', VOICE_TYPES.get(voice_type, VOICE_TYPES["default"])["volume"])
    
    tts_engine.say(text)
    tts_engine.runAndWait()

# 2. Initialize the local AI model with Ollama using Qwen2.5-0.5B-Instruct
llm = ChatOllama(
    model="qwen2.5:0.5b-instruct",
    temperature=0.3,
)

# 3. Define native live web search tool
@tool
def live_web_search(query: str) -> str:
    """Searches the live web for current events, news, or factual data using DuckDuckGo."""
    try:
        print(f"\n[Robot System]: Searching the web for '{query}'...")
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No relevant live search results found."
            formatted = "\n".join([f"- {r['title']}: {r['body']} ({r['href']})" for r in results])
            return formatted
    except Exception as e:
        return f"Live search failed due to an error: {str(e)}"

# 4. Define the ad-blocked YouTube audio streaming tool
@tool
def play_youtube_audio(query: str) -> str:
    """Searches YouTube for a song or video and streams audio locally, auto-skipping ads and sponsorships via SponsorBlock."""
    try:
        print(f"\n[Robot System]: Searching YouTube (Ad-blocked) for '{query}'...")
        cmd = [
            "yt-dlp", 
            f"ytsearch1:{query}", 
            "--get-url", 
            "--get-title",
            "--sponsorblock-remove", "all"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return "Could not find a matching track on YouTube."
        
        title = lines[0]
        stream_url = lines[1]
        
        print(f"[Robot System]: Now playing ad-free: {title}")
        subprocess.Popen(["mpv", "--no-video", stream_url])
        
        return f"Successfully started playing '{title}' from YouTube with ads blocked."
    except Exception as e:
        return f"Failed to play YouTube audio due to an error: {str(e)}"

# 5. Bind tools directly to the model
tools = [live_web_search, play_youtube_audio]
llm_with_tools = llm.bind_tools(tools)

# Global variables for selected persona and voice
selected_persona = "default"
selected_voice = "default"

def get_selected_persona():
    """Return the current persona system prompt"""
    return PERSONAS.get(selected_persona, PERSONAS["default"])

def set_persona_and_voice(persona, voice):
    """Set the global persona and voice settings"""
    global selected_persona, selected_voice
    selected_persona = persona
    selected_voice = voice

def create_ui():
    """Create a simple UI for selecting persona and voice type"""
    root = tk.Tk()
    root.title("Robo-Qwen Configuration")
    root.geometry("400x300")
    
    # Title
    title_label = tk.Label(root, text="Robo-Qwen Assistant", font=("Arial", 16, "bold"))
    title_label.pack(pady=10)
    
    # Persona selection
    persona_frame = tk.Frame(root)
    persona_frame.pack(pady=10)
    
    tk.Label(persona_frame, text="Select Persona:", font=("Arial", 12)).pack()
    
    persona_var = tk.StringVar(value="default")
    persona_combo = ttk.Combobox(persona_frame, textvariable=persona_var, 
                                 values=list(PERSONAS.keys()), state="readonly", width=20)
    persona_combo.pack(pady=5)
    
    # Voice type selection
    voice_frame = tk.Frame(root)
    voice_frame.pack(pady=10)
    
    tk.Label(voice_frame, text="Select Voice Type:", font=("Arial", 12)).pack()
    
    voice_var = tk.StringVar(value="default")
    voice_combo = ttk.Combobox(voice_frame, textvariable=voice_var, 
                               values=list(VOICE_TYPES.keys()), state="readonly", width=20)
    voice_combo.pack(pady=5)
    
    # Apply button
    def apply_settings():
        set_persona_and_voice(persona_var.get(), voice_var.get())
        root.destroy()
    
    apply_button = tk.Button(root, text="Apply Settings", command=apply_settings, 
                           font=("Arial", 12), bg="#4CAF50", fg="white", width=15)
    apply_button.pack(pady=20)
    
    # Start button
    def start_assistant():
        set_persona_and_voice(persona_var.get(), voice_var.get())
        root.destroy()
    
    start_button = tk.Button(root, text="Start Assistant", command=start_assistant, 
                           font=("Arial", 12), bg="#2196F3", fg="white", width=15)
    start_button.pack(pady=5)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

def robot_interact(user_query: str):
    system_prompt = get_selected_persona()
    system_message = SystemMessage(content=system_prompt)
    
    messages = [system_message, HumanMessage(content=user_query)]
    
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            selected_tool = {
                "live_web_search": live_web_search,
                "play_youtube_audio": play_youtube_audio
            }.get(tool_call["name"])
            
            if selected_tool:
                print(f"[Robot System]: Executing tool '{tool_call['name']}'...")
                tool_output = selected_tool.invoke(tool_call["args"])
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
        
        final_response = llm_with_tools.invoke(messages)
        return final_response.content
    
    return ai_msg.content

def listen_to_microphone() -> str:
    """Listens to the microphone and converts speech to text using Google's speech recognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[Robot Ears]: Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("[Robot Ears]: Processing audio...")
            text = recognizer.recognize_google(audio)
            print(f"You (Voice): {text}")
            return text
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("[Robot Ears]: Could not understand audio.")
            return ""
        except Exception as e:
            print(f"[Robot Ears Error]: {str(e)}")
            return ""

def main():
    # Show configuration UI first
    create_ui()
    
    greeting = "Robo-Qwen Assistant voice interface online. Speak into your microphone or type 'exit' to quit."
    print(greeting)
    speak(greeting, selected_voice)
    print("=" * 50)
    
    while True:
        try:
            mode = input("\nPress [Enter] to speak, or type your query (or 'exit'): ").strip()
            
            if mode.lower() in ['quit', 'exit', 'q']:
                farewell = "Shutting down systems. Goodbye!"
                print(f"\nRobo-Qwen: {farewell}")
                speak(farewell, selected_voice)
                break
            
            if mode == "":
                user_input = listen_to_microphone()
                if not user_input:
                    continue
            else:
                user_input = mode
                
            response = robot_interact(user_input)
            print(f"\nRobo-Qwen: {response}")
            speak(response, selected_voice)
            
        except KeyboardInterrupt:
            print("\n\nRobo-Qwen: Emergency stop triggered. Goodbye!")
            break
        except Exception as e:
            err_msg = f"Error processing request: {str(e)}"
            print(f"\nRobo-Qwen: {err_msg}")
            speak("An internal system error occurred.", selected_voice)

if __name__ == "__main__":
    main()
