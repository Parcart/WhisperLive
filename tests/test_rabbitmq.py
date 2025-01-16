import aio_pika
import asyncio
import uuid
import json
import base64
from typing import Optional

import numpy as np


class RabbitMQManager:
    def __init__(self, rabbitmq_host="localhost", rabbitmq_port=5672, rabbitmq_user="guest", rabbitmq_password="guest"):
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.rabbitmq_user = rabbitmq_user
        self.rabbitmq_password = rabbitmq_password
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.futures = {}

    async def connect(self):
        self.connection = await aio_pika.connect_robust(
            host=self.rabbitmq_host,
            port=self.rabbitmq_port,
            login=self.rabbitmq_user,
            password=self.rabbitmq_password
        )

        self.channel = await self.connection.channel()
        # Создаем временную очередь для ответов
        result = await self.channel.declare_queue(name='', exclusive=True)
        self.callback_queue = result.name
        await result.consume(self.on_response)

    async def close(self):
        if self.channel:
            await self.channel.close()
        if self.connection:
            await self.connection.close()

    async def on_response(self, message: aio_pika.IncomingMessage):
        async with message.process():
            correlation_id = message.correlation_id
            if correlation_id and correlation_id in self.futures:
                future = self.futures.pop(correlation_id)
                data = json.loads(message.body)
                future.set_result(data)

    async def send_and_wait_for_response(self, audio_bytes: bytes, routing_key='audio_queue') -> Optional[dict]:
        try:
            future = asyncio.get_running_loop().create_future()
            correlation_id = str(uuid.uuid4())

            self.futures[correlation_id] = future

            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=audio_bytes,
                    correlation_id=correlation_id,
                    reply_to=self.callback_queue,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=routing_key
            )

            print(f"Audio message sent with correlation_id: {correlation_id}, waiting for response...")
            return await future

        except Exception as e:
            print(f"Failed to send message or get a response from RabbitMQ: {e}")
            return None


def read_file(filename='test_woman_resampled16000.wav'):
    with open(filename, "rb") as f:
        data = f.read()

    raw_data = np.frombuffer(buffer=data, dtype=np.int16)
    audio_array = raw_data.astype(np.float32) / 32768.0
    audio_bytes = audio_array.tobytes()

    return audio_bytes


async def main():
    manager = RabbitMQManager(rabbitmq_host="localhost", rabbitmq_port=5672)
    await manager.connect()

    response = await manager.send_and_wait_for_response(read_file())

    if response:
        print("Received response:", response)
    else:
        print("No response received.")

    await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
