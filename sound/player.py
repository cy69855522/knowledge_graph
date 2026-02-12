import sys
import os
import asyncio
import threading
import tempfile
import yaml
import markdown
import pygame
import nest_asyncio
from mutagen.mp3 import MP3
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QFileDialog, 
                             QTextBrowser, QLabel, QProgressBar)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
import edge_tts

# Patch asyncio for mixed threaded/async environments
nest_asyncio.apply()

class AudioWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(float)
    
    def __init__(self, text, voice, rate, volume):
        super().__init__()
        self.text = text
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.temp_file = None

    async def _generate_audio(self):
        communicate = edge_tts.Communicate(self.text, self.voice, rate=self.rate, volume=self.volume)
        self.temp_file = tempfile.mktemp(suffix=".mp3")
        await communicate.save(self.temp_file)

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._generate_audio())
        self.finished.emit()

class TTSPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Infra Guide - TTS Player 🗣️")
        self.setGeometry(100, 100, 1000, 800)
        
        # Load Config
        self.load_config()
        
        # Init Pygame Mixer
        pygame.mixer.init()
        
        # UI Setup
        self.init_ui()
        
        # Timer for Progress Update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        
        self.audio_file = None
        self.is_playing = False
        self.is_paused = False
        self.duration = 0

    def load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {"tts": {"voice": "zh-CN-XiaoxiaoNeural", "rate": "+0%", "volume": "+0%"}}

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Top Bar: File Selection
        top_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        select_btn = QPushButton("Open Markdown File 📂")
        select_btn.clicked.connect(self.open_file)
        top_layout.addWidget(select_btn)
        top_layout.addWidget(self.file_label)
        layout.addLayout(top_layout)
        
        # Middle: Markdown Viewer
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        layout.addWidget(self.text_browser)
        
        # Bottom: Player Controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Generate & Play ▶️")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Stop ⏹️")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 1000) # 0 to 100.0%
        self.progress_slider.sliderPressed.connect(self.slider_pressed)
        self.progress_slider.sliderReleased.connect(self.slider_released)
        
        self.time_label = QLabel("00:00 / 00:00")
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.progress_slider)
        controls_layout.addWidget(self.time_label)
        
        layout.addLayout(controls_layout)
        
        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Markdown", "", "Markdown Files (*.md)")
        if file_path:
            self.stop_audio()
            self.current_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            with open(file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
                html = markdown.markdown(md_text, extensions=['fenced_code', 'tables'])
                # Basic CSS for better look
                css = """
                <style>
                    body { font-family: sans-serif; line-height: 1.6; padding: 20px; color: #333; }
                    h1, h2, h3 { color: #2c3e50; }
                    code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 4px; font-family: monospace; }
                    pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }
                    blockquote { border-left: 4px solid #ddd; padding-left: 10px; color: #666; }
                </style>
                """
                self.text_browser.setHtml(css + html)
                # Strip markdown for TTS (simple approach)
                self.plain_text = self.strip_markdown(md_text)
                self.play_btn.setEnabled(True)
                self.status_bar.showMessage("File loaded. Click Play to generate audio.")

    def strip_markdown(self, text):
        # Very basic stripping, assuming the user triggers clean read
        # In a real app, use BeautifulSoup on the HTML output
        from bs4 import BeautifulSoup
        html = markdown.markdown(text)
        return "".join(BeautifulSoup(html, "html.parser").findAll(text=True))

    def toggle_play(self):
        if self.is_playing:
            self.pause_audio()
        elif self.is_paused:
            self.resume_audio()
        else:
            self.generate_and_play()

    def generate_and_play(self):
        self.status_bar.showMessage("Generating Audio... (This requires internet for edge-tts)")
        self.play_btn.setEnabled(False)
        self.worker = AudioWorker(self.plain_text, 
                                  self.config['tts']['voice'], 
                                  self.config['tts']['rate'], 
                                  self.config['tts']['volume'])
        self.worker_thread = threading.Thread(target=self.worker.run)
        self.worker.finished.connect(self.on_audio_generated)
        self.worker_thread.start()

    def on_audio_generated(self):
        self.audio_file = self.worker.temp_file

        # Get duration using mutagen
        try:
            audio = MP3(self.audio_file)
            self.duration = audio.info.length
        except Exception as e:
            print(f"Error getting duration: {e}")
            self.duration = 0

        pygame.mixer.music.load(self.audio_file)
        pygame.mixer.music.play()
        self.is_playing = True
        self.status_bar.showMessage("Playing 🔊")
        self.play_btn.setText("Pause ⏸️")
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        self.progress_slider.setRange(0, int(self.duration))
        self.timer.start(500) 

    def pause_audio(self):
        pygame.mixer.music.pause()
        self.is_playing = False
        self.is_paused = True
        self.play_btn.setText("Resume ▶️")
        self.status_bar.showMessage("Paused ⏸️")

    def resume_audio(self):
        pygame.mixer.music.unpause()
        self.is_playing = True
        self.is_paused = False
        self.play_btn.setText("Pause ⏸️")
        self.status_bar.showMessage("Playing 🔊")
    
    def stop_audio(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False
        self.play_btn.setText("Generate & Play ▶️")
        self.stop_btn.setEnabled(False)
        self.timer.stop()
        self.progress_slider.setValue(0)
        self.status_bar.showMessage("Stopped ⏹️")

    def update_progress(self):
        if pygame.mixer.music.get_busy() or self.is_playing:
            # pygame.mixer.music.get_pos() returns ms
            current_ms = pygame.mixer.music.get_pos()
            # It resets after pause/unpause on some systems, so this is a simplified view
            # Robust seek/pause implementation requires tracking offset manually
            seconds = int(current_ms / 1000)
            
            # Update slider if not being dragged
            if not self.progress_slider.isSliderDown():
                self.progress_slider.setValue(seconds)
            
            total_seconds = int(self.duration)
            self.time_label.setText(f"{seconds // 60:02d}:{seconds % 60:02d} / {total_seconds // 60:02d}:{total_seconds % 60:02d}")
        else:
            if self.is_playing and not self.is_paused: 
                # Finished naturally
                self.stop_audio()

    def slider_pressed(self):
        self.timer.stop()

    def slider_released(self):
        # Seeking is hard with pygame.mixer.music without precise duration
        # Leaving as TBD for V2
        self.timer.start(500)
        pass

    def closeEvent(self, event):
        pygame.mixer.quit()
        if self.audio_file and os.path.exists(self.audio_file):
            try:
                os.remove(self.audio_file)
            except:
                pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TTSPlayer()
    window.show()
    sys.exit(app.exec())
