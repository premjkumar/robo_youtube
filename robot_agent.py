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

# 1. Initialize Text-to-Speech (Robot Voice)
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 165)
tts_engine.setProperty('volume', 1.0)

def speak(text: str):
    """Speaks the response out loud using the local system TTS engine."""
    print(f"\n[Robot Voice]: {text}")
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

# 6. System prompt defining the robot persona
SYSTEM_PROMPT = """You are an autonomous robotic companion running locally on Fedora Linux powered by Qwen. 
You have direct tool access to live web search and ad-blocked YouTube audio playback. 
Keep your responses sharp, logical, concise, and characteristic of an advanced synthetic unit since they will be spoken aloud."""

system_message = SystemMessage(content=SYSTEM_PROMPT)

def robot_interact(user_query: str):
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
    greeting = "Robo-Qwen Assistant voice interface online. Speak into your microphone or type 'exit' to quit."
    print(greeting)
    speak(greeting)
    print("=" * 50)
    
    while True:
        try:
            mode = input("\nPress [Enter] to speak, or type your query (or 'exit'): ").strip()
            
            if mode.lower() in ['quit', 'exit', 'q']:
                farewell = "Shutting down systems. Goodbye!"
                print(f"\nRobo-Qwen: {farewell}")
                speak(farewell)
                break
            
            if mode == "":
                user_input = listen_to_microphone()
                if not user_input:
                    continue
            else:
                user_input = mode
                
            response = robot_interact(user_input)
            print(f"\nRobo-Qwen: {response}")
            speak(response)
            
        except KeyboardInterrupt:
            print("\n\nRobo-Qwen: Emergency stop triggered. Goodbye!")
            break
        except Exception as e:
            err_msg = f"Error processing request: {str(e)}"
            print(f"\nRobo-Qwen: {err_msg}")
            speak("An internal system error occurred.")

if __name__ == "__main__":
    main()