import logging
import threading
import time

import pika
import json

from .client import TranscriptionClient


class RabbitMQConsumer:
    def __init__(self, rabbitmq_host="localhost", rabbitmq_port=5672, whisper_host="localhost", whisper_port=5000):
        self.rabbitmq_host = rabbitmq_host
        self.rabbitmq_port = rabbitmq_port
        self.whisper_host = whisper_host
        self.whisper_port = whisper_port
        self.connection = None
        self.channel = None
        self.thread = None
        self.is_running = False

    def connect(self):
        try:
            credentials = pika.PlainCredentials('guest', 'guest')
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(self.rabbitmq_host, self.rabbitmq_port, credentials=credentials))
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue='audio_queue', durable=True)
            self.channel.queue_declare(queue='stt_queue', durable=True)
        except Exception as e:
            print(f"Error connecting to RabbitMQ: {e}")
            time.sleep(3)
            print("Retrying...")
            self.connect()

    def disconnect(self):
        if self.channel and self.channel.is_open:
            self.channel.close()
        if self.connection and self.connection.is_open:
            self.connection.close()

    def __publish(self, properties, body):
        self.channel.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            body=body,
            properties=pika.BasicProperties(
                correlation_id=properties.correlation_id,
                delivery_mode=2
            )
        )

    def _callback(self, ch, method, properties, body):
        """Обработка сообщений."""
        print("Полученное аудио")
        try:
            if isinstance(body, bytes):
                stt_text = self._process_stt(body)
                self.__publish(properties, json.dumps({"result": stt_text}))
                print("STT ответ отправлен")
            else:
                raise ValueError("Полученное сообщение не является байтовым массивом")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"Ошибка Callback: {e}")
            self.__publish(properties, json.dumps({"error": str(e)}))
            # Логируем ошибку, но не возвращаем сообщение в очередь
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def _process_stt(self, audio_bytes) -> str:
        """Имитация STT Processing"""
        logging.info("[CLIENT] Processing STT")
        print("INFO: Processing STT")
        client = TranscriptionClient(
            self.whisper_host,
            self.whisper_port,
            lang=None,
            translate=False,
            model="large-v2",
            use_vad=False,
        )
        try:
            client(audio_data=audio_bytes)
        except Exception as e:
            logging.info(f"Error processing STT: {e}")
            print(f"Error processing STT: {e}")

        result = "".join([transcript["text"] for transcript in client.clients[0].transcript])

        logging.info(f"[CLIENT] Result: {result}")

        return result

    def run(self):
        """Запускаем Consuming в отдельном потоке"""
        self.is_running = True
        self.thread = threading.Thread(target=self._run_consumer,
                                       daemon=True)
        self.thread.start()
        return self

    def _run_consumer(self):
        """Функция для запуска потребителя."""
        try:
            self.connect()
            self.channel.basic_qos(prefetch_count=1)
            self.channel.basic_consume(queue='audio_queue', on_message_callback=self._callback)
            print("RabbitMQConsumer Started")
            self.channel.start_consuming()
        except Exception as e:
            print(f"Ошибка Consumer: {e}")
        finally:
            self.disconnect()

    def stop_consuming(self):
        """Останавливает потребление."""
        self.is_running = False
        if self.channel and self.channel.is_open:
            self.channel.stop_consuming()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.disconnect()


if __name__ == "__main__":
    consumer = RabbitMQConsumer()
    consumer.run()
    consumer.thread.join()
