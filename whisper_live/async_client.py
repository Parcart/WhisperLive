import asyncio
import os
import shutil
import subprocess
import wave

import logging
from typing import AsyncIterable, AsyncGenerator

import numpy as np
import pyaudio
import threading
import json
import uuid
import time
import ffmpeg

from websockets.asyncio.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

import whisper_live.utils as utils


class AsyncClient:
    """
    Handles communication with a server using WebSocket.
    """
    INSTANCES = {}
    END_OF_AUDIO = "END_OF_AUDIO"

    def __init__(
            self,
            host=None,
            port=None,
            lang=None,
            translate=False,
            model="small",
            srt_file_path="output.srt",
            use_vad=True,
            log_transcription=True,
            eventloop=None
    ):
        """
        Initializes a Client instance for audio recording and streaming to a server.

        If host and port are not provided, the WebSocket connection will not be established.
        When translate is True, the task will be set to "translate" instead of "transcribe".
        he audio recording starts immediately upon initialization.

        Args:
            host (str): The hostname or IP address of the server.
            port (int): The port number for the WebSocket server.
            lang (str, optional): The selected language for transcription. Default is None.
            translate (bool, optional): Specifies if the task is translation. Default is False.
        """
        self.recording = False
        self.task = "transcribe"
        self.uid = str(uuid.uuid4())
        self.waiting = False
        self.last_response_received = None
        self.disconnect_if_no_response_for = 15
        self.language = lang
        self.model = model
        self.server_error = False
        self.srt_file_path = srt_file_path
        self.use_vad = use_vad
        self.last_segment = None
        self.last_received_segment = None
        self.log_transcription = log_transcription
        self.ws_connected = False
        self.client_socket = None
        self.eventloop = eventloop

        if translate:
            self.task = "translate"

        self.audio_bytes = None

        if self.eventloop is None:
            self.eventloop = asyncio.new_event_loop()

        print("[INFO]: Running client...")
        if host is not None and port is not None:
            socket_url = f"ws://{host}:{port}"
            self.eventloop.create_task(self.run_websocket(socket_url))
        else:
            print("[ERROR]: No host or port specified.")
            return

        AsyncClient.INSTANCES[self.uid] = self

        self.transcript = []

    async def await_run_websocket(self):
        while True:
            if not self.ws_connected:
                await asyncio.sleep(0.1)
            else:
                break

    async def run_websocket(self, socket_url):
        try:
            async with connect(socket_url) as websocket:
                self.client_socket: ClientConnection = websocket
                await self.on_open(websocket)
                await self.websocket_recv()
        except Exception as e:
            self.on_error(e)

    async def websocket_recv(self):
        try:
            while True:
                message = await self.client_socket.recv(decode=True)
                await self.on_message(message)
        except ConnectionClosed as e:
            self.on_close(e.rcvd.code, e.rcvd.reason)

    def handle_status_messages(self, message_data):
        """Handles server status messages."""
        status = message_data["status"]
        if status == "WAIT":
            self.waiting = True
            print(f"[INFO]: Server is full. Estimated wait time {round(message_data['message'])} minutes.")
        elif status == "ERROR":
            print(f"Message from Server: {message_data['message']}")
            self.server_error = True
        elif status == "WARNING":
            print(f"Message from Server: {message_data['message']}")

    def process_segments(self, segments):
        """Processes transcript segments."""
        text = []
        for i, seg in enumerate(segments):
            if not text or text[-1] != seg["text"]:
                text.append(seg["text"])
                if i == len(segments) - 1:
                    self.last_segment = seg
                elif (self.server_backend == "faster_whisper" and
                      (not self.transcript or
                       float(seg['start']) >= float(self.transcript[-1]['end']))):
                    self.transcript.append(seg)
        # update last received segment and last valid response time
        if self.last_received_segment is None or self.last_received_segment != segments[-1]["text"]:
            self.last_response_received = time.time()
            self.last_received_segment = segments[-1]["text"]

        if self.log_transcription:
            # Truncate to last 3 entries for brevity.
            utils.clear_screen()
            utils.print_transcript(text)

    async def on_message(self, message):
        """
        Callback function called when a message is received from the server.

        It updates various attributes of the client based on the received message, including
        recording status, language detection, and server messages. If a disconnect message
        is received, it sets the recording status to False.

        Args:
            ws (websockets.asyncio.client.ClientConnection): The WebSocket client instance.
        """
        message = json.loads(message)

        if self.uid != message.get("uid"):
            print("[ERROR]: invalid client uid")
            return

        if "status" in message.keys():
            self.handle_status_messages(message)
            return

        if "message" in message.keys() and message["message"] == "DISCONNECT":
            print("[INFO]: Server disconnected due to overtime.")
            self.recording = False

        if "message" in message.keys() and message["message"] == "SERVER_READY":
            self.last_response_received = time.time()
            self.recording = True
            self.server_backend = message["backend"]
            print(f"[INFO]: Server Running with backend {self.server_backend}")
            return

        if "language" in message.keys():
            self.language = message.get("language")
            lang_prob = message.get("language_prob")
            print(
                f"[INFO]: Server detected language {self.language} with probability {lang_prob}"
            )
            return

        if "segments" in message.keys():
            self.process_segments(message["segments"])

    def on_error(self, error):
        print(f"[ERROR] WebSocket Error: {error}")
        self.server_error = True
        self.error_message = error

    def on_close(self, close_status_code, close_msg):
        print(f"[INFO]: Websocket connection closed: {close_status_code}: {close_msg}")
        self.recording = False
        self.waiting = False
        self.ws_connected = False

    async def on_open(self, ws):
        """
        Callback function called when the WebSocket connection is successfully opened.

        Sends an initial configuration message to the server, including client UID,
        language selection, and task type.

        Args:
            ws (websockets.ClientConnection): The WebSocket client instance.

        """
        print("[INFO]: Opened connection")
        await ws.send(
            json.dumps(
                {
                    "uid": self.uid,
                    "language": self.language,
                    "task": self.task,
                    "model": self.model,
                    "use_vad": self.use_vad
                }
            )
        )
        self.ws_connected = True

    async def send_packet_to_server(self, message):
        """
        Send an audio packet to the server using WebSocket.

        Args:
            message (bytes): The audio data packet in bytes to be sent to the server.

        """
        try:
            await self.client_socket.send(message)
        except Exception as e:
            print(e)
            raise e

    async def close_websocket(self):
        """
        Close the WebSocket connection and join the WebSocket thread.

        First attempts to close the WebSocket connection using `self.client_socket.close()`. After
        closing the connection, it joins the WebSocket thread to ensure proper termination.

        """
        try:
            await self.client_socket.close()
        except Exception as e:
            print("[ERROR]: Error closing WebSocket:", e)

    def get_client_socket(self):
        """
        Get the WebSocket client socket instance.

        Returns:
            WebSocketApp: The WebSocket client socket instance currently in use by the client.
        """
        return self.client_socket

    def write_srt_file(self, output_path="output.srt"):
        """
        Writes out the transcript in .srt format.

        Args:
            message (output_path, optional): The path to the target file.  Default is "output.srt".

        """
        if self.server_backend == "faster_whisper":
            if (self.last_segment):
                self.transcript.append(self.last_segment)
            utils.create_srt_file(self.transcript, output_path)

    async def wait_before_disconnect(self):
        """Waits a bit before disconnecting in order to process pending responses."""
        assert self.last_response_received
        # (time.time() - self.last_response_received < self.disconnect_if_no_response_for) and
        while self.ws_connected:
            await asyncio.sleep(0.1)
        pass


class TranscriptionTeeClient:

    def __init__(self, clients, eventloop, save_output_recording=False,
                 output_recording_filename="./output_recording.wav"):
        self.clients = clients
        if not self.clients:
            raise Exception("At least one client is required.")
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 60000
        self.save_output_recording = save_output_recording
        self.output_recording_filename = output_recording_filename
        self.frames = b""
        self.p = pyaudio.PyAudio()
        self.eventloop = eventloop
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
        except OSError as error:
            print(f"[WARN]: Unable to access microphone. {error}")
            self.stream = None

    async def __call__(self, audio=None, async_audio_generator=None):
        """
        Start the transcription process.

        Initiates the transcription process by connecting to the server via a WebSocket. It waits for the server
        to be ready to receive audio data and then sends audio for transcription. If an audio file is provided, it
        will be played and streamed to the server; otherwise, it will perform live recording.

        Args:
            audio (str, optional): Path to an audio file for transcription. Default is None, which triggers live recording.

        """
        assert sum(
            source is not None for source in
            [audio, async_audio_generator]
        ) <= 1, 'You must provide only one selected source'

        await self.await_server_ready()

        print("[INFO]: Server Ready!")
        # if hls_url is not None:
        #     self.process_hls_stream(hls_url, save_file)
        if audio is not None:
            resampled_file = audio
            # resampled_file = utils.resample(audio)
            await self.play_file(resampled_file)
        # elif rtsp_url is not None:
        #     self.process_rtsp_stream(rtsp_url)
        # elif file_async_generator is not None:
        #     self.eventloop.run_until_complete(self.process_async_generator(file_async_generator))
        # elif pcm_generator is not None:
        #     self.eventloop.run_until_complete(self.stream_file_send(pcm_generator))
        elif async_audio_generator is not None:
            await self.streaming_audio(async_audio_generator)
        else:
            raise

    async def await_server_ready(self):
        print("[INFO]: Waiting for server ready ...")
        for client in self.clients:
            # recording = True -> server is ready
            while not client.recording:
                await asyncio.sleep(0.1)
                if client.waiting or client.server_error:
                    await self.close_all_clients()
                    raise

    async def close_all_clients(self):
        """Closes all client websockets."""
        for client in self.clients:
            await client.close_websocket()

    def write_all_clients_srt(self):
        """Writes out .srt files for all clients."""
        for client in self.clients:
            client.write_srt_file(client.srt_file_path)

    async def multicast_packet(self, packet, unconditional=False):
        """
        Sends an identical packet via all clients.

        Args:
            packet (bytes): The audio data packet in bytes to be sent.
            unconditional (bool, optional): If true, send regardless of whether clients are recording.  Default is False.
        """
        for client in self.clients:
            if (unconditional or client.recording):
                await client.send_packet_to_server(packet)

    async def play_file(self, filename):
        """
        Play an audio file and send it to the server for processing.

        Reads an audio file, plays it through the audio output, and simultaneously sends
        the audio data to the server for processing. It uses PyAudio to create an audio
        stream for playback. The audio data is read from the file in chunks, converted to
        floating-point format, and sent to the server using WebSocket communication.
        This method is typically used when you want to process pre-recorded audio and send it
        to the server in real-time.

        Args:
            filename (str): The path to the audio file to be played and sent to the server.
        """

        # read audio and create pyaudio stream
        with wave.open(filename, "rb") as wavfile:
            self.stream = self.p.open(
                format=self.p.get_format_from_width(wavfile.getsampwidth()),
                channels=wavfile.getnchannels(),
                rate=wavfile.getframerate(),
                input=True,
                output=True,
                frames_per_buffer=self.chunk,
            )

            try:
                while any(client.recording for client in self.clients):
                    data = wavfile.readframes(self.chunk)
                    if data == b"":
                        break

                    audio_array = self.bytes_to_float_array(data)
                    await self.multicast_packet(audio_array.tobytes())
                    self.stream.write(data)

                wavfile.close()
                start_time = time.time()

                await self.multicast_packet(AsyncClient.END_OF_AUDIO.encode('utf-8'), True)
                print("[INFO]: Sending end of audio.")

                for client in self.clients:
                    await client.wait_before_disconnect()

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Время выполнения команды : {elapsed_time:.4f} секунд")

                self.write_all_clients_srt()
                self.stream.close()
                await self.close_all_clients()

            except KeyboardInterrupt:
                wavfile.close()
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()
                await self.close_all_clients()
                self.write_all_clients_srt()
                print("[INFO]: Keyboard interrupt.")

    async def streaming_audio(self, request_iterator: AsyncGenerator[bytes, None]):
        try:
            while any(client.recording for client in self.clients):
                async for data in request_iterator:
                    if data == b"":
                        break

                    audio_array = self.bytes_to_float_array(data)
                    await self.multicast_packet(audio_array.tobytes())
                    # self.stream.write(data)
                break

            start_time = time.time()
            await self.multicast_packet(AsyncClient.END_OF_AUDIO.encode('utf-8'), True)
            print("[INFO]: Sending end of audio.")

            for client in self.clients:
                await client.wait_before_disconnect()
            self.write_all_clients_srt()
            await self.close_all_clients()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Время обработки последнего сегмента : {elapsed_time:.4f} секунд")
            return

        except KeyboardInterrupt:
            await self.close_all_clients()
            self.write_all_clients_srt()
            print("[INFO]: Keyboard interrupt.")

    def send_file(self, filename):

        # read audio and create pyaudio stream
        with wave.open(filename, "rb") as wavfile:
            try:
                data = b""
                while any(client.recording for client in self.clients):
                    chunk_data = wavfile.readframes(self.chunk)
                    if chunk_data == b"":
                        break
                    data += chunk_data

                audio_array = self.bytes_to_float_array(data)
                self.multicast_packet(audio_array.tobytes())

                wavfile.close()
                start_time = time.time()

                self.multicast_packet(AsyncClient.END_OF_AUDIO.encode('utf-8'), True)

                for client in self.clients:
                    client.wait_before_disconnect()

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Время выполнения команды : {elapsed_time:.4f} секунд")

                self.write_all_clients_srt()
                self.close_all_clients()

            except KeyboardInterrupt:
                wavfile.close()
                self.close_all_clients()
                self.write_all_clients_srt()
                print("[INFO]: Keyboard interrupt.")

    def write_audio_frames_to_file(self, frames, file_name):
        """
        Write audio frames to a WAV file.

        The WAV file is created or overwritten with the specified name. The audio frames should be
        in the correct format and match the specified channel, sample width, and sample rate.

        Args:
            frames (bytes): The audio frames to be written to the file.
            file_name (str): The name of the WAV file to which the frames will be written.

        """
        with wave.open(file_name, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            wavfile.writeframes(frames)

    @staticmethod
    def bytes_to_float_array(audio_bytes):
        """
        Convert audio data from bytes to a NumPy float array.

        It assumes that the audio data is in 16-bit PCM format. The audio data is normalized to
        have values between -1 and 1.

        Args:
            audio_bytes (bytes): Audio data in bytes.

        Returns:
            np.ndarray: A NumPy array containing the audio data as float values normalized between -1 and 1.
        """
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0


class AsyncTranscriptionClient(TranscriptionTeeClient):
    """
    Client for handling audio transcription tasks via a single WebSocket connection.

    Acts as a high-level client for audio transcription tasks using a WebSocket connection. It can be used
    to send audio data for transcription to a server and receive transcribed text segments.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port number to connect to on the server.
        lang (str, optional): The primary language for transcription. Default is None, which defaults to English ('en').
        translate (bool, optional): Indicates whether translation tasks are required (default is False).
        save_output_recording (bool, optional): Indicates whether to save recording from microphone.
        output_recording_filename (str, optional): File to save the output recording.
        output_transcription_path (str, optional): File to save the output transcription.

    Attributes:
        client (Client): An instance of the underlying Client class responsible for handling the WebSocket connection.

    Example:
        To create a TranscriptionClient and start transcription on microphone audio:
        ```python
        transcription_client = TranscriptionClient(host="localhost", port=9090)
        transcription_client()
        ```
    """

    def __init__(
            self,
            host,
            port,
            lang=None,
            translate=False,
            model="small",
            use_vad=True,
            save_output_recording=False,
            output_recording_filename="./output_recording.wav",
            output_transcription_path="./output.srt",
            log_transcription=True,
            eventloop=None
    ):
        if eventloop is None:
            eventloop = asyncio.get_event_loop()
        self.client = AsyncClient(host, port, lang, translate, model, srt_file_path=output_transcription_path,
                                  use_vad=use_vad, log_transcription=log_transcription, eventloop=eventloop)
        if save_output_recording and not output_recording_filename.endswith(".wav"):
            raise ValueError(f"Please provide a valid `output_recording_filename`: {output_recording_filename}")
        if not output_transcription_path.endswith(".srt"):
            raise ValueError(
                f"Please provide a valid `output_transcription_path`: {output_transcription_path}. The file extension should be `.srt`.")
        TranscriptionTeeClient.__init__(
            self,
            [self.client],
            save_output_recording=save_output_recording,
            output_recording_filename=output_recording_filename,
            eventloop=eventloop
        )
