import logging
import cv2
import asyncio
import aiohttp 
import telegram
from telegram.ext import Application
from configparser import ConfigParser

class NotificationManager:
    def __init__(self, config: ConfigParser):
        self.api_endpoint = config.get('API', 'endpoint', fallback=None)
        self.telegram_token = config.get('Telegram', 'token', fallback=None)
        self.telegram_chat_id = config.get('Telegram', 'chat_id', fallback=None)
        
        self.http_session = None

        self.bot = None
        if self.telegram_token and self.telegram_chat_id:
            try:
                application = Application.builder().token(self.telegram_token).pool_timeout(120).get_updates_pool_timeout(120).build()
                self.bot = application.bot
            except Exception as e:
                logging.error(f"Failed to initialize Telegram bot: {e}")
    
    async def start_session(self):
        if self.api_endpoint:
            self.http_session = aiohttp.ClientSession()

    async def close_session(self):
        if self.http_session:
            await self.http_session.close()
    
    async def send_api_request(self, plate: str, status: str):
        if not self.http_session or not plate:
            return
        try:
            url = f"{self.api_endpoint}/{plate}/{status}"
            async with self.http_session.get(url, timeout=5) as response:
                if response.status != 200:
                    logging.warning(f"API request to {url} returned the status: {response.status}")
        except Exception as e:
            logging.error(f"API request error: {e}")

    async def send_telegram_photo(self, plate: str, status: str, image_np):
        if not self.bot:
            return
        
        try:
            caption = f"Detected number: {plate}\Status: {status}"
            
            success, encoded_image = cv2.imencode('.jpg', image_np)
            if not success:
                logging.error("Couldn't encode the image")
                return

            await self.bot.send_photo(
                chat_id=self.telegram_chat_id,
                photo=encoded_image.tobytes(),
                caption=caption
            )
        except Exception as e:
            logging.error(f"Error when sending the photo: {e}")