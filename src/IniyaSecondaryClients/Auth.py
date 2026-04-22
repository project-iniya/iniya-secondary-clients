from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import requests
import webbrowser
import keyring
import os

from .utils import get_device_id

class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, state=None, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)

        if "code" in query:
            code = query["code"][0]

            res = requests.post(
                "https://iniyaai-backend.onrender.com/api/auth/exchange-code",
                json={"code": code}
            )

            data = res.json()
            self.state["token"] = data.get("token")
            self.state["devid"] = data.get("devid")

            self.send_response(302)
            self.send_header("Location", "https://iniya-ai.vercel.app/successLogin")
            self.end_headers()
        
    
def login(provider: str):
    DEVICE_PATH = os.path.expanduser("~/.iniya/device_id")
    os.makedirs(os.path.dirname(DEVICE_PATH), exist_ok=True)

    if not provider:
        raise Exception("Provider not specified")

    state = {
        "token": None,
        "devid": None
    }

    def handler(*args, **kwargs):
        Handler(*args, state=state, **kwargs)

    server = HTTPServer(("localhost", 3001), handler)

    webbrowser.open(f"https://iniyaai-backend.onrender.com/api/auth/{provider}")
    print("Waiting for login...")

    server.handle_request()

    TOKEN = state["token"]
    DEVID = state["devid"]

    if not TOKEN or not DEVID:
        raise Exception("Login failed: missing token or device id")

    with open(DEVICE_PATH, "w") as f:
        f.write(DEVID)

    os.chmod(DEVICE_PATH, 0o600)

    keyring.set_password("IniyaAI", DEVID, TOKEN)

    return DEVID
    
def verify_token(token: str):
    res = requests.post(
        "https://iniyaai-backend.onrender.com/api/auth/verify-token",
        json={"token": token, "devid": get_device_id()}
    )
    data = res.json()

    if data.get('valid'):
        return True

def logout():
    devid = get_device_id()
    DEVICE_PATH = os.path.expanduser("~/.iniya/device_id")

    try:
        token = keyring.get_password("IniyaAI", devid)

        res = requests.post(
            "https://iniyaai-backend.onrender.com/api/auth/logout",
            headers={"Authorization":f"Bearer {token}"},
            json={"devid": devid}
        )

        data = res.json()

        if data.get('ok') :
            with open(DEVICE_PATH, 'w') as f:
                f.write("")
              
            keyring.delete_password("IniyaAI", devid)
        else:
            raise Exception("Failed To logout", {data.get('error')})
    except Exception as e:
        print(e)
