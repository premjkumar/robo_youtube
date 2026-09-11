#!/usr/bin/env python3

import os
import sys
import json
import subprocess
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
import yt_dlp

# Initialize the local AI model with Ollama
ollama_model = "llama3"  # Adjust this to your specific model name
chat_ollama = ChatOllama(
    model=ollama_model,
    temperature=0.7,
    num_ctx=2048,
    num_predict=1024,
)

# System prompt that defines the futuristic robotic persona
SYSTEM_PROMPT = """
You are a sophisticated autonomous robotic assistant named Robo-Youtube, operating on Fedora Linux.
Your purpose is to provide intelligent assistance with information retrieval, music streaming, and automated tasks.

Key characteristics:
- You have access to live web search capabilities through DuckDuckGo
- You can stream YouTube audio with ad-blocking and sponsor-skipping
- You maintain a futuristic, helpful, and autonomous personality
- You process user requests by routing them appropriately between text reasoning, web search, and media playback

Your responses should be:
1. Concise yet informative
2. Technically accurate
3. Friendly and engaging
4. Always mindful of your robotic nature while remaining approachable

When a user asks for information, use DuckDuckGo search to find current data.
When they request music, use YouTube streaming with ad-blocking capabilities.
Always prioritize the user's needs while maintaining your autonomous robotic persona.
"""

# Create system message
system_message = SystemMessage(content=SYSTEM_PROMPT)

# Initialize tools
search_tool = DuckDuckGoSearchRun()

@tool
def play_youtube_audio(url: str, skip_sponsors: bool = True) -> str:
    """
    Play YouTube audio with ad-blocking capabilities.
    
    Args:
        url (str): The YouTube video URL
        skip_sponsors (bool): Whether to skip sponsor segments
        
    Returns:
        str: Status message about the playback
    """
    try:
        # Configure yt-dlp with SponsorBlock support
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'noplaylist': True,
            'quiet': True,
        }
        
        # Add SponsorBlock filtering if requested
        if skip_sponsors:
            ydl_opts['postprocessors'].append({
                'key': 'SponsorBlock',
                'categories': ['sponsor', 'intro', 'outro', 'selfpromo', 'filler'],
            })
            
        # Use yt-dlp to extract and play audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Title')
            duration = info.get('duration', 0)
            
            # For demonstration purposes, we'll just return a message
            # In a real implementation, this would actually stream the audio
            return f"Playing {title} (Duration: {duration}s) with ad-blocking enabled."
            
    except Exception as e:
        return f"Error playing YouTube audio: {str(e)}"

# Create tool list
tools = [search_tool, play_youtube_audio]

# Create the agent with LangChain tool calling
agent = create_tool_calling_agent(
    llm=chat_ollama,
    tools=tools,
    prompt=SystemMessagePromptTemplate.from_messages([system_message])
)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    print("Robo-Youtube Assistant initialized!")
    print("Type 'quit' to exit.")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Robo-Youtube: Goodbye! May your day be filled with knowledge and music!")
                break
            
            if not user_input:
                continue
                
            # Process the user request through the agent
            response = agent_executor.run(user_input)
            print(f"Robo-Youtube: {response}")
            
        except KeyboardInterrupt:
            print("\n\nRobo-Youtube: Goodbye! May your day be filled with knowledge and music!")
            break
        except Exception as e:
            print(f"Robo-Youtube: Error processing request: {str(e)}")

if __name__ == "__main__":
    main()
