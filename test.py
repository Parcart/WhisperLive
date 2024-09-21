import asyncio
import multiprocessing
import time

from whisper_live.client import TranscriptionClient
import whisper_live.utils as utils

client = TranscriptionClient(
    "localhost",
    9090,
    lang=None,
    translate=False,
    model="large-v2",
    use_vad=False
)


# client()

async def readfile(filename):
    with open(filename, "rb") as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            yield data


# async def omg():
#     async for data in readfile("tests/test_woman"):
#         print(data)
#
# asyncio.run(omg())




# asyncio.run(client(file_async_generator=readfile(resampled_file)))
def send_file(command):
    start_time = time.time()
    current_loop = asyncio.get_event_loop()
    resampled_file = utils.resample("tests/test_1.ogg")
    client(pcm_generator=readfile(resampled_file), event_loop=current_loop)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Время выполнения команды '{command}': {elapsed_time:.4f} секунд")


def stream_file(command):
    start_time = time.time()
    client("tests/test_1.ogg")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Время выполнения команды '{command}': {elapsed_time:.4f} секунд")


stream_file("1")

# send_file("2")

# if __name__ == '__main__':
#     with multiprocessing.Pool(processes=4) as pool:
#         pool.map(stream_file, ["1", "2", "3", "4"])

