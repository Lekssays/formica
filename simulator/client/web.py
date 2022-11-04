import websockets
import asyncio
import datetime


async def send_log(message: str):
    uri = "ws://172.17.0.1:7777"
    # dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    message = "hello"
    print("1sent message")
    async with websockets.connect(uri) as websocket:
        print("2sent message")
        await websocket.send(message)
        print("3sent message")


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_log("hello"))


if __name__ == '__main__':
    main()