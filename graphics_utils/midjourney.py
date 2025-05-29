import json
import os
import random
import time
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# See here for an explanation of where these values come from:
# https://github.com/yachty66/unofficial_midjourney_python_api
MJ_APPLICATION_ID = os.getenv("MJ_APPLICATION_ID")
MJ_GUILD_ID = os.getenv("MJ_GUILD_ID")
MJ_CHANNEL_ID = os.getenv("MJ_CHANNEL_ID")
MJ_VERSION = os.getenv("MJ_VERSION")
MJ_ID = os.getenv("MJ_ID")
MJ_AUTHORIZATION = os.getenv("MJ_AUTHORIZATION")

# Validate that all required environment variables are set
required_env_vars = {
    "MJ_APPLICATION_ID": MJ_APPLICATION_ID,
    "MJ_GUILD_ID": MJ_GUILD_ID,
    "MJ_CHANNEL_ID": MJ_CHANNEL_ID,
    "MJ_VERSION": MJ_VERSION,
    "MJ_ID": MJ_ID,
    "MJ_AUTHORIZATION": MJ_AUTHORIZATION,
}

missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file.")

class MidjourneyApi():
    def __init__(self, prompt, application_id, guild_id, channel_id, version, id, authorization, download_dir, file_name):
        self.application_id = application_id
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.version = version
        self.id = id
        self.authorization = authorization
        self.prompt = prompt
        self.download_dir = download_dir
        self.file_name = file_name
        self.message_id = ""
        self.custom_id = ""
        self.image_path_str = ""
        self.send_message()
        self.get_message()
        self.choose_images()
        self.download_image()

    def send_message(self):
        url = "https://discord.com/api/v9/interactions"
        data = {
            "type": 2,
            "application_id": self.application_id,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "session_id": "cannot be empty",
            "data": {
                "version": self.version,
                "id": self.id,
                "name": "imagine",
                "type": 1,
                "options": [
                    {
                        "type": 3,
                        "name": "prompt",
                        "value": self.prompt
                    }
                ],
                "application_command": {
                    "id": self.id,
                    "application_id": self.application_id,
                    "version": self.version,
                    "default_member_permissions": None,
                    "type": 1,
                    "nsfw": False,
                    "name": "imagine",
                    "description": "Create images with Midjourney",
                    "dm_permission": True,
                    "contexts": None,
                    "options": [
                        {
                            "type": 3,
                            "name": "prompt",
                            "description": "The prompt to imagine",
                            "required": True
                        }
                    ]
                },
                "attachments": []
            },
        }
        headers = {
            'Authorization': self.authorization,
            'Content-Type': 'application/json',
        }
        # print("Sending message", headers, data)
        response = requests.post(url, headers=headers, json=data)

    def get_message(self):
        headers = {
            'Authorization': self.authorization,
            "Content-Type": "application/json",
        }
        for i in range(3):
            time.sleep(random.uniform(25, 35))
            try:
                response = requests.get(f'https://discord.com/api/v9/channels/{self.channel_id}/messages', headers=headers)
                messages = response.json()
                most_recent_message_id = messages[0]['id']
                # print("Messages", messages)
                self.message_id = most_recent_message_id
                components = messages[0]['components'][0]['components']
                buttons = [comp for comp in components if comp.get('label') in ['U1', 'U2', 'U3', 'U4']]
                custom_ids = [button['custom_id'] for button in buttons]
                random_custom_id = random.choice(custom_ids)
                self.custom_id = random_custom_id
                break
            except Exception as e:
                # Exceptions parsing the messages happen when it's not done generating yet
                # print("Exception getting message", e)
                ValueError("Timeout")

    def choose_images(self):
        url = "https://discord.com/api/v9/interactions"
        headers = {
            "Authorization": self.authorization,
            "Content-Type": "application/json",
        }
        data = {
            "type": 3,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "message_flags": 0,
            "message_id": self.message_id,
            "application_id": self.application_id,
            "session_id": "cannot be empty",
            "data": {
                "component_type": 2,
                "custom_id": self.custom_id,
            }
        }
        # print("Choosing images", headers, data)
        response = requests.post(url, headers=headers, data=json.dumps(data))

    def download_image(self):
        headers = {
            'Authorization': self.authorization,
            "Content-Type": "application/json",
        }
        for i in range(10):
            time.sleep(random.uniform(10, 20))
            try:
                response = requests.get(f'https://discord.com/api/v9/channels/{self.channel_id}/messages', headers=headers)
                messages = response.json()
                most_recent_message_id = messages[0]['id']
                # print("Downloading image", messages)
                self.message_id = most_recent_message_id
                image_url = messages[0]['attachments'][0]['url']
                image_response = requests.get(image_url)
                a = urlparse(image_url)
                image_name = self.file_name
                if not image_name.endswith(".png"):
                    image_name = image_name + ".png"
                self.image_path_str = f"{self.download_dir}/{image_name}"
                if not os.path.exists(self.download_dir):
                    os.makedirs(self.download_dir, exist_ok=True)
                with open(f"{self.download_dir}/{image_name}", "wb") as file:
                    file.write(image_response.content)
                break
            except Exception as e:
                print("Got error downloading image:", e)
                raise ValueError("Timeout")

    def image_path(self):
        return self.image_path_str

def generate_image_and_save_to_file(prompt: str, download_path: str = "images", aspect_ratio: str = " --ar 4:3"):
    prompt = prompt + aspect_ratio

    download_dir, file_name = os.path.split(download_path)

    midjourney = MidjourneyApi(prompt=prompt, application_id=MJ_APPLICATION_ID, guild_id=MJ_GUILD_ID, channel_id=MJ_CHANNEL_ID, version=MJ_VERSION, id=MJ_ID, authorization=MJ_AUTHORIZATION, download_dir=download_dir, file_name=file_name)

    return midjourney.image_path()


if __name__ == "__main__":
    # midjourney = MidjourneyApi(prompt="Awesome wizard battle but both wizards are birds", application_id=MJ_APPLICATION_ID, guild_id=MJ_GUILD_ID, channel_id=MJ_CHANNEL_ID, version=MJ_VERSION, id=MJ_ID, authorization=MJ_AUTHORIZATION, download_dir="images", file_name="test.png")

    generate_image_and_save_to_file(prompt="Awesome wizard battle but both wizards are birds", download_path="images/test.png")

    # print("Final image:", midjourney.image_path())
