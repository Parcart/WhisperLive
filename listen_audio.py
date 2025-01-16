import pyaudio


class ListenAudio:
    format_audio = pyaudio.paInt16
    channels = 1
    rate = 24000
    chunk = 2048

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format_audio,
                                      channels=self.channels,
                                      rate=16000,
                                      input=True,
                                      output=True,
                                      frames_per_buffer=2048)

    def listen(self, filename):
        with open(filename, 'rb') as f:
            while True:
                data = f.read(self.chunk)
                if not data:
                    break
                self.stream.write(data)

        self.stream.close()


if __name__ == '__main__':
    listen = ListenAudio()
    listen.listen('Бу_испугался_16')
