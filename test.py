import asyncio
import multiprocessing
import time
import wave
# import aiofiles
import aiofiles
import pyaudio

from whisper_live.async_client import AsyncTranscriptionClient as TranscriptionClient
import whisper_live.utils as utils
result = []

# client = TranscriptionClient(
#     "localhost",
#     9090,
#     lang=None,
#     translate=False,
#     model="large-v2",
#     use_vad=False
# )
async def async_audio_stream_generator(filename):
    chunk = 4096
    try:
        async with aiofiles.open(filename, "rb") as wavfile:
            start_time = time.time()
            # with wave.open(filename, "rb") as wavfile:
            try:
                while True:
                    # print("INFO: streaming file", filename)
                    data = await wavfile.read(chunk)
                    if data == b"":
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"INFO Время чтения файла {filename}: {elapsed_time:.4f} секунд")
                        break

                    await asyncio.sleep(0.11)

                    yield data
            except Exception as e:
                print(e)
                pass
            finally:
                print("INFO: Closing file", filename)
                await wavfile.close()
    except Exception as e:
        print(e)


# client()

async def readfile(filename):
    with open(filename, "rb") as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            yield data


async def async_FileStreamSTT(client, filename, name):
    start_time = time.time()
    await client(filename)
    result = "".join([transcript["text"] for transcript in client.clients[0].transcript])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Время выполнения команды '{name}': {elapsed_time:.4f} секунд")
    print("Результат выполнения STT: " + result)


async def async_AudioStreamSTT(client, filename, name):
    start_time = time.time()
    request_iterator = async_audio_stream_generator(filename)
    await client(async_audio_generator=request_iterator)
    result_stt = "".join([transcript["text"] for transcript in client.clients[0].transcript])
    end_time = time.time()
    elapsed_time = end_time - start_time
    complete_time = f"Время выполнения команды '{name}': {elapsed_time:.4f} секунд"
    result_stt_str = f"Результат выполнения STT: {result_stt}"
    result.append(complete_time)
    result.append(result_stt_str)
    print(complete_time)
    print(result_stt_str)


async def create_and_run_tasks():
    tasks = list()
    client = TranscriptionClient(
        "localhost",
        9090,
        lang=None,
        translate=False,
        model="large-v2",
        use_vad=False,
        eventloop=asyncio.get_running_loop()
    )
    tasks.append(
        asyncio.create_task(async_AudioStreamSTT(client, "tests/test_woman_resampled16000.wav", "Test1"), name="Test1"))
    # client2 = TranscriptionClient(
    #     "213.181.122.2",
    #     40054,
    #     lang=None,
    #     translate=False,
    #     model="large-v2",
    #     use_vad=False,
    #     eventloop=asyncio.get_running_loop()
    # )
    # tasks.append(
    #     asyncio.create_task(async_AudioStreamSTT(client2, "tests/test_1_resampled.wav", "Test2"), name="Test2"))

    result = asyncio.gather(*tasks)
    await result
    [print(r) for r in result]


async def main():
    # await create_and_run_tasks()
    result = asyncio.gather(create_and_run_tasks())
    await result
    pass


if __name__ == '__main__':
    # with open("tests/test_woman_resampled41000.wav", "rb") as f:
    #     data = f.read()
    asyncio.run(main())

#     with multiprocessing.Pool(processes=4) as pool:
#         pool.map(stream_file, ["1", "2", "3", "4"])

from faster_whisper.utils import download_model
