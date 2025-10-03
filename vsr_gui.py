import sys
import os
import signal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                           QFileDialog, QCheckBox, QGroupBox, QMessageBox,
                           QProgressBar, QComboBox, QTextEdit)
from PyQt6.QtCore import Qt, QProcess, QRegularExpression

class FileDropWidget(QWidget):
    def __init__(self, placeholder_text="Drop file here or click to browse", parent=None):
        super().__init__(parent)
        self.placeholder_text = placeholder_text
        self.setAcceptDrops(True)
        self.file_path = ""
        
        self.layout = QVBoxLayout(self)
        self.label = QLabel(self.placeholder_text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)
        
        self.setStyleSheet("""
            FileDropWidget {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 10px;
                min-height: 60px;
            }
            FileDropWidget:hover {
                border-color: #555;
            }
        """)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.file_path = files[0]
            self.update_label()
    
    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            self.file_path = file_path
            self.update_label()
            
    def update_label(self):
        if self.file_path:
            self.label.setText(os.path.basename(self.file_path))
        else:
            self.label.setText(self.placeholder_text)
            
    def get_file_path(self):
        return self.file_path
    
    def set_file_path(self, path):
        self.file_path = path
        self.update_label()


class VSRGUIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VSR GUI")
        self.setMinimumWidth(600)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Model type selection
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["PyTorch (.pth)", "ONNX (.onnx)"])
        self.model_type_combo.currentIndexChanged.connect(self.on_model_type_changed)
        model_type_layout.addWidget(self.model_type_combo)
        model_type_layout.addStretch(1)
        main_layout.addLayout(model_type_layout)
        
        # Input video section
        input_group = QGroupBox("Input Video")
        input_layout = QVBoxLayout()
        self.input_widget = FileDropWidget("Drop video file here or click to browse")
        input_layout.addWidget(self.input_widget)
        input_group.setLayout(input_layout)
        
        # Model selection section
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("Select a model file (.pth or .onnx)")
        model_browse_btn = QPushButton("Browse")
        model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_path_edit, 3)
        model_layout.addWidget(model_browse_btn, 1)
        model_group.setLayout(model_layout)
        
        # Output video section
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Specify output video path")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_path_edit, 3)
        output_layout.addWidget(output_browse_btn, 1)
        output_group.setLayout(output_layout)
        
        # Options section
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()

        codec_layout = QHBoxLayout()
        codec_layout.addWidget(QLabel("Video Codec:"))
        
        self.codec_combo = QComboBox()
        # Add common codec options
        codec_options = [
            "libx264",     # CPU H.264
            "libx265",     # CPU H.265
            "dnxhd",       # DNxHD (high quality)
            "prores",      # ProRes (high quality)
            "Custom..."    # Allow custom codec input
        ]
        self.codec_combo.addItems(codec_options)
        self.codec_combo.setCurrentText("h264_nvenc")
        self.codec_combo.currentTextChanged.connect(self.on_codec_changed)
        codec_layout.addWidget(self.codec_combo)
        
        # Custom codec input that appears when "Custom..." is selected
        self.custom_codec_input = QLineEdit()
        self.custom_codec_input.setPlaceholderText("Enter custom codec")
        self.custom_codec_input.setVisible(False)
        codec_layout.addWidget(self.custom_codec_input)
        
        options_layout.addLayout(codec_layout)
        options_group.setLayout(options_layout)
        
        # Process button and Stop button in horizontal layout
        button_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Process Video")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setStyleSheet("font-weight: bold; height: 30px;")
        
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setStyleSheet("height: 30px;")
        self.stop_btn.setToolTip("Forcibly stop the processing. This will result in an incomplete output file.")
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.stop_btn)
        
        # Progress bar and FPS in horizontal layout
        progress_layout = QHBoxLayout()
        
        # Progress bar (left side)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                height: 25px;
                min-height: 25px;
                margin: 0px;
                padding: 0px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;  /* Green color */
                width: 10px;
                margin: 0px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # FPS label (right side)
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setMinimumWidth(100)  # Give it some minimum width
        progress_layout.addWidget(self.fps_label)
        
        # Log box
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(150)
        log_layout.addWidget(self.log_box)
        log_group.setLayout(log_layout)
        
        # Add everything to main layout
        main_layout.addWidget(input_group)
        main_layout.addWidget(model_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(options_group)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(progress_layout)  # Add the combined progress/FPS layout
        main_layout.addWidget(log_group)
        
        # Initialize QProcess
        self.process = None
        self.total_frames = 0
        self.current_frame = 0
        
        self.setCentralWidget(main_widget)
        
    def on_model_type_changed(self, index):
        # Update placeholder text and file filter based on selection
        if index == 0:  # PyTorch
            self.model_path_edit.setPlaceholderText("Select a PyTorch model file (.pth)")
        else:  # ONNX
            self.model_path_edit.setPlaceholderText("Select an ONNX model file (.onnx)")
        
        # Clear current selection
        self.model_path_edit.clear()
        
    def browse_model(self):
        if self.model_type_combo.currentIndex() == 0:  # PyTorch
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select PyTorch Model File", "", "PyTorch Models (*.pth)"
            )
        else:  # ONNX
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select ONNX Model File", "", "ONNX Models (*.onnx)"
            )
            
        if file_path:
            self.model_path_edit.setText(file_path)
            
    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video", "", "Video Files (*.mkv)"
        )
        if file_path:
            if not file_path.endswith('.mkv'):
                file_path += '.mkv'
            self.output_path_edit.setText(file_path)
    
    def process_video(self):
        # Validate inputs
        input_path = self.input_widget.get_file_path()
        model_path = self.model_path_edit.text()
        output_path = self.output_path_edit.text()
        
        if not input_path:
            QMessageBox.warning(self, "Missing Input", "Please select an input video file.")
            return
        
        if not model_path:
            QMessageBox.warning(self, "Missing Model", "Please select a model file.")
            return
        
        if not output_path:
            QMessageBox.warning(self, "Missing Output", "Please specify an output path.")
            return
        
        # Determine which codec to use
        if self.codec_combo.currentText() == "Custom...":
            codec = self.custom_codec_input.text()
            if not codec:
                QMessageBox.warning(self, "Missing Codec", "Please enter a custom codec.")
                return
        else:
            codec = self.codec_combo.currentText()
        
        # Determine which script to run based on model type
        if self.model_type_combo.currentIndex() == 0:  # PyTorch
            script_name = "test_vsr.py"
        else:  # ONNX
            script_name = "test_onnx.py"
        
        # Build command
        cmd = [
            "python", script_name,
            "--model_path", model_path,
            "--input", input_path,
            "--output", output_path,
            "--video", codec,
            "--gui-mode"
        ]
        
        # Clear previous log
        self.log_box.clear()
        
        # Reset frame counters
        self.total_frames = 0
        self.current_frame = 0
        
        # Show progress bar with determinate mode initially
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Disable process button, enable stop button
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Create and start QProcess
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)
        
        try:
            self.log_box.append("Starting video processing...")
            self.log_box.append(f"Command: {' '.join(cmd)}\n")
            self.process.start(cmd[0], cmd[1:])
            
        except Exception as e:
            self.log_box.append(f"Error: {str(e)}")
            self.process_finished()
            
    def stop_processing(self):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.log_box.append("\nStopping process...")
            
            try:
                # Terminate the process gracefully
                self.process.terminate()
                
                # Give it a few seconds to terminate gracefully
                if not self.process.waitForFinished(3000):
                    # If it didn't terminate, kill it forcefully
                    self.log_box.append("Process did not terminate gracefully, forcing kill...")
                    self.process.kill()
                    
            except Exception as e:
                self.log_box.append(f"Error while stopping process: {str(e)}")
                # If we can't terminate, try to kill it
                try:
                    self.process.kill()
                except:
                    pass
    
    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode()
        
        # Check for our special GUI progress format
        if 'PROGRESS:' in stdout:
            # Split the line into progress and FPS parts
            parts = stdout.split('PROGRESS:')[1].strip().split('|')
            
            # Handle progress
            if len(parts) >= 1:
                progress_parts = parts[0].split('/')
                if len(progress_parts) == 2:
                    self.current_frame = int(progress_parts[0])
                    
                    # If this is the first time we're getting the total
                    if self.total_frames == 0:
                        self.total_frames = int(progress_parts[1])
                        # Now that we know the total, set the progress bar maximum
                        self.progress_bar.setMaximum(self.total_frames)
                    
                    # Update progress bar
                    self.progress_bar.setValue(self.current_frame)
            
            # Handle FPS
            if len(parts) >= 2 and 'FPS:' in parts[1]:
                try:
                    fps = float(parts[1].split(':')[1])
                    self.fps_label.setText(f"FPS: {fps:.2f}")
                except (IndexError, ValueError):
                    pass
        
        # Always append to log
        self.log_box.append(stdout)
        
    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode()
        self.log_box.append(stderr)
        
    def process_finished(self, exit_code=None, exit_status=None):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fps_label.setText("FPS: --")  # Reset FPS display
        self.log_box.append("\nProcessing completed!")
        self.process = None

    def on_codec_changed(self, text):
        if text == "Custom...":
            self.custom_codec_input.setVisible(True)
        else:
            self.custom_codec_input.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VSRGUIApp()
    window.show()
    sys.exit(app.exec())
