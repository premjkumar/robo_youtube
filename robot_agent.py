import threading
import queue
import time
import tkinter as tk
from tkinter import scrolledtext, ttk
import subprocess
import tempfile
import os
import speech_recognition as sr

from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# --- LangGraph Assistant Core ---
class VoiceAssistantCore:
    def __init__(self, model_name="qwen"):
        try:
            self.llm = Ollama(model=model_name)
            
            workflow = StateGraph(state_schema=MessagesState)
            
            def call_model(state: MessagesState):
                messages = state["messages"]
                prompt_history = "\n".join([
                    f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
                    for m in messages[:-1]
                ])
                current_input = messages[-1].content
                
                full_prompt = f"{prompt_history}\nUser: {current_input}\nAssistant:" if prompt_history else f"User: {current_input}\nAssistant:"
                response_text = self.llm.invoke(full_prompt)
                return {"messages": [AIMessage(content=response_text.strip())]}

            workflow.add_node("model", call_model)
            workflow.add_edge(START, "model")
            workflow.add_edge("model", END)

            self.memory = MemorySaver()
            self.graph = workflow.compile(checkpointer=self.memory)
            self.thread_config = {"configurable": {"thread_id": "desktop_voice_thread_1"}}
            
        except Exception as e:
            self.llm = None
            self.graph = None
            print(f"Graph Initialization Error: {e}")

        self.model_path = "en_US-lessac-medium.onnx"

    def get_response(self, prompt):
        if not self.graph:
            return "Error: LangGraph or Ollama model is not initialized."
        try:
            input_message = HumanMessage(content=prompt)
            output = self.graph.invoke({"messages": [input_message]}, self.thread_config)
            return output["messages"][-1].content
        except Exception as e:
            return f"Generation Error: {str(e)}"

    def speak(self, text):
        try:
            speech_file = tempfile.mktemp(suffix=".wav")
            cmd = f"echo '{text}' | piper --model {self.model_path} --output_file {speech_file}"
            subprocess.run(cmd, shell=True, check=True)
            subprocess.run(f"mpv {speech_file} --no-video --really-quiet", shell=True)
            if os.path.exists(speech_file):
                os.remove(speech_file)
        except Exception as e:
            print(f"Speech Error: {e}")

# --- Modern Tkinter UI ---
class VoiceAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robo-Qwen Voice Assistant")
        self.root.geometry("750x580")
        self.root.minsize(550, 420)

        self.assistant = VoiceAssistantCore()
        self.task_queue = queue.Queue()
        self.recognizer = sr.Recognizer()
        
        # Adjust energy threshold to better handle normal speaking volume
        self.recognizer.energy_threshold = 300
        self.mic_index = None  # Set integer index if your default mic fails

        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.bg_color = "#1e1e1e"
        self.fg_color = "#d4d4d4"
        self.accent_color = "#007acc"
        self.mic_color = "#28a745"
        
        self.root.configure(bg=self.bg_color)
        self.create_widgets()
        self.check_queue()

    def create_widgets(self):
        title_label = tk.Label(
            self.root, 
            text="Robo-Qwen Voice Assistant (LangGraph + Piper TTS)", 
            font=("Segoe UI", 14, "bold"),
            bg=self.bg_color, 
            fg="#ffffff"
        )
        title_label.pack(pady=10)

        chat_frame = tk.Frame(self.root, bg=self.bg_color)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            state='disabled',
            font=("Segoe UI", 11),
            bg="#252526",
            fg=self.fg_color,
            insertbackground="white",
            borderwidth=0,
            highlightthickness=0
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            font=("Segoe UI", 9, "italic"),
            bg=self.bg_color, 
            fg="#858585",
            anchor="w"
        )
        status_bar.pack(fill=tk.X, padx=15, pady=2)

        input_frame = tk.Frame(self.root, bg=self.bg_color)
        input_frame.pack(fill=tk.X, padx=15, pady=12)

        self.input_field = tk.Entry(
            input_frame, 
            font=("Segoe UI", 12),
            bg="#333333",
            fg="#ffffff",
            insertbackground="white",
            relief=tk.FLAT
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, ipadx=5)
        self.input_field.bind("<Return>", lambda event: self.on_send_text())

        self.mic_button = tk.Button(
            input_frame, 
            text="🎤 Speak", 
            command=self.on_mic_click,
            font=("Segoe UI", 10, "bold"),
            bg=self.mic_color,
            fg="white",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            cursor="hand2"
        )
        self.mic_button.pack(side=tk.RIGHT, padx=(8, 0))

        self.send_button = tk.Button(
            input_frame, 
            text="Send", 
            command=self.on_send_text,
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_color,
            fg="white",
            relief=tk.FLAT,
            padx=15,
            pady=5,
            cursor="hand2"
        )
        self.send_button.pack(side=tk.RIGHT, padx=(8, 0))

    def append_chat(self, sender, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: ", "bold")
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.tag_config("bold", foreground="#4ec9b0", font=("Segoe UI", 11, "bold"))
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)

    def on_send_text(self):
        user_text = self.input_field.get().strip()
        if not user_text:
            return
        self.input_field.delete(0, tk.END)
        self.process_user_query(user_text)

    def on_mic_click(self):
        self.lock_inputs()
        self.status_var.set("Listening... Speak now (up to 30s)")
        threading.Thread(target=self.listen_audio, daemon=True).start()

    def listen_audio(self):
        try:
            # Buffer to let audio channel fully close from previous speech playback
            time.sleep(0.8)

            mic_kwargs = {}
            if self.mic_index is not None:
                mic_kwargs["device_index"] = self.mic_index

            with sr.Microphone(**mic_kwargs) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                # Expanded phrase_time_limit to 30 seconds for longer inputs/song names
                audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=30.0)
            
            self.status_var.set("Processing voice input...")
            query = self.recognizer.recognize_google(audio)
            self.task_queue.put(("user_input", query))
        except sr.WaitTimeoutError:
            self.status_var.set("Ready (No speech detected)")
            self.task_queue.put(("idle", ""))
        except sr.UnknownValueError:
            self.status_var.set("Ready (Could not understand audio)")
            self.task_queue.put(("idle", ""))
        except Exception as e:
            print(f"Microphone Error Details: {e}")
            self.status_var.set("Ready")
            self.task_queue.put(("idle", ""))

    def process_user_query(self, query):
        self.lock_inputs()
        self.append_chat("You", query)
        self.status_var.set("LangGraph node processing...")
        threading.Thread(target=self.run_assistant_pipeline, args=(query,), daemon=True).start()

    def run_assistant_pipeline(self, query):
        response_text = self.assistant.get_response(query)
        self.task_queue.put(("response", response_text))
        self.assistant.speak(response_text)
        self.task_queue.put(("idle", ""))

    def lock_inputs(self):
        self.input_field.config(state='disabled')
        self.send_button.config(state='disabled')
        self.mic_button.config(state='disabled')

    def check_queue(self):
        try:
            while True:
                task_type, data = self.task_queue.get_nowait()
                if task_type == "user_input":
                    self.process_user_query(data)
                elif task_type == "response":
                    self.append_chat("Assistant", data)
                elif task_type == "idle":
                    self.input_field.config(state='normal')
                    self.send_button.config(state='normal')
                    self.mic_button.config(state='normal')
                    self.status_var.set("Ready")
                    self.input_field.focus()
                self.task_queue.task_done()
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAssistantApp(root)
    root.mainloop()