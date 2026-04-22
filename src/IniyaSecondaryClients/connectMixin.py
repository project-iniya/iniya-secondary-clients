"""
ConnectMixin — connects the Iniya SDK to A Moblie Controller

IN-DEVELOPMENT: Using This Will Not be Possible 

"""

import asyncio
import websockets
import json
import webbrowser
import threading
import requests
from typing import Callable, Optional


class ConnectMixin:

    def __init__(self):
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._session_id: Optional[str] = None
        self._connected: bool = False
        self._client_connected: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._server_url: Optional[str] = None  # base HTTP url e.g. https://iniya.onrender.com

        # Callbacks
        self._on_connect_cb: Optional[Callable] = None
        self._on_disconnect_cb: Optional[Callable] = None
        self._on_data_cb: Optional[Callable[[dict], None]] = None      # generic JSON from phone
        self._on_audio_cb: Optional[Callable[[bytes], None]] = None    # raw audio bytes
        self._on_frame_cb: Optional[Callable[[bytes], None]] = None    # raw image/frame bytes
        self._on_error_cb: Optional[Callable[[Exception], None]] = None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def connect(self, server_url: str = "https://iniyaai-backend.onrender.com/api", open_browser: bool = True) -> dict:
        """
        Create a session on the Iniya server and connect via WebSocket.

        Args:
            server_url: Base HTTP URL e.g. "https://iniyaai-backend.onrender.com"
            open_browser: Auto-open the session page (with QR) in browser.

        Returns:
            dict with keys: session_id, hex_code, session_url, qr_base64
        
        Raises:
            RuntimeError: if session creation fails.
        """
        if self._connected:
            raise RuntimeError("[ConnectMixin] Already connected. Call disconnect() first.")

        self._server_url = server_url.rstrip("/")

        # 1. Ask server to create a session
        session = self._create_session()
        self._session_id = session["sessionId"]

        # 2. Open the browser so the phone can scan the QR
        if open_browser:
            webbrowser.open(session["sessionConnectLink"])

        # 3. Start WS in background thread
        self._loop = asyncio.new_event_loop()
        self._ws_thread = threading.Thread(
            target=self._run_ws_loop,
            daemon=True
        )
        self._ws_thread.start()

        print(f"[ConnectMixin] Session ready. hex_code: {session['hex_code']}")
        print(f"[ConnectMixin] Waiting for phone to connect...")
        return session

    def disconnect(self):
        """
        Disconnect from the server and clean up all state.
        """
        if not self._connected:
            print("[ConnectMixin] Not connected.")
            return

        self._connected = False
        self._client_connected = False

        if self._ws and self._loop:
            future = asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._ws = None
        self._session_id = None
        print("[ConnectMixin] Disconnected from server.")

    def send(self, data: dict):
        """
        Send a JSON message to the phone via the server bridge.
        Use sparingly — phone drives the session, but you may need to
        send acks, config, or control signals back.

        Args:
            data: dict, will be sent as JSON.
        
        Raises:
            RuntimeError: if not connected or no phone is in the session.
        """
        if not self._connected:
            raise RuntimeError("[ConnectMixin] Not connected to server.")
        if not self._client_connected:
            raise RuntimeError("[ConnectMixin] No phone connected to this session yet.")

        asyncio.run_coroutine_threadsafe(
            self._ws.send(json.dumps(data)), self._loop
        )

    def is_client_connected(self) -> bool:
        """Returns True if a phone is currently in the session."""
        return self._client_connected

    # ──────────────────────────────────────────────
    # Callback Decorators
    # ──────────────────────────────────────────────

    def on_connect(self, func: Callable):
        """Called when the phone joins the session."""
        self._on_connect_cb = func
        return func

    def on_disconnect(self, func: Callable):
        """Called when the phone leaves the session."""
        self._on_disconnect_cb = func
        return func

    def on_data(self, func: Callable[[dict], None]):
        """
        Called when the phone sends a generic JSON payload.
        Covers anything that isn't raw audio/frame bytes —
        sensor data, gestures, text, custom events, etc.

        Args:
            func: fn(data: dict)
        """
        self._on_data_cb = func
        return func

    def on_audio(self, func: Callable[[bytes], None]):
        """
        Called when the phone sends an audio chunk.

        Args:
            func: fn(audio_bytes: bytes)
        """
        self._on_audio_cb = func
        return func

    def on_frame(self, func: Callable[[bytes], None]):
        """
        Called when the phone sends a camera frame.

        Args:
            func: fn(frame_bytes: bytes)
        """
        self._on_frame_cb = func
        return func

    def on_error(self, func: Callable[[Exception], None]):
        """
        Called on any connection-level error.

        Args:
            func: fn(error: Exception)
        """
        self._on_error_cb = func
        return func

    # ──────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────

    def _create_session(self) -> dict:
        """
        POST /connect/create on the server.
        Expects response: { session_id, hex_code, session_url, qr_base64 }
        """
        try:
            resp = requests.post(f"{self._server_url}/connect/create", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"[ConnectMixin] Failed to create session: {e}")

    def _run_ws_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_handler())

    async def _ws_handler(self):
        """
        Connects to the server WSS endpoint for this session.
        Reconnects automatically if the connection drops (Render cold starts etc.)
        """
        wss_base = self._server_url.replace("https://", "wss://").replace("http://", "ws://")
        uri = f"{wss_base}/connect/sdk/{self._session_id}"

        while self._connected or not self._ws:  # allow first connection
            try:
                async with websockets.connect(uri) as ws:
                    self._ws = ws
                    self._connected = True
                    print("[ConnectMixin] WSS connected to server.")

                    async for message in ws:
                        await self._route_message(message)

            except websockets.ConnectionClosed:
                if not self._connected:
                    break  # intentional disconnect, stop retrying
                print("[ConnectMixin] Connection dropped. Reconnecting in 3s...")
                await asyncio.sleep(3)

            except Exception as e:
                if self._on_error_cb:
                    self._on_error_cb(e)
                else:
                    print(f"[ConnectMixin] WS error: {e}")
                await asyncio.sleep(3)

    async def _route_message(self, message: str | bytes):
        """
        Routes incoming messages from the server to the right callback.

        Protocol (server decides the format):
          - JSON string  → event messages (client_connected, client_disconnected) or on_data payload
          - bytes        → binary data from phone; server prefixes 1 byte:
                           0x01 = audio chunk
                           0x02 = camera frame
        """
        if isinstance(message, bytes):
            if len(message) < 2:
                return
            msg_type, payload = message[0], message[1:]
            if msg_type == 0x01 and self._on_audio_cb:
                self._on_audio_cb(payload)
            elif msg_type == 0x02 and self._on_frame_cb:
                self._on_frame_cb(payload)
            return

        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return

        event = msg.get("event")

        if event == "client_connected":
            self._client_connected = True
            print("[ConnectMixin] Phone connected to session.")
            if self._on_connect_cb:
                self._on_connect_cb()

        elif event == "client_disconnected":
            self._client_connected = False
            print("[ConnectMixin] Phone left session.")
            if self._on_disconnect_cb:
                self._on_disconnect_cb()

        elif event == "data" and self._on_data_cb:
            # Phone sent a structured JSON payload; server wraps it as { event: "data", payload: {...} }
            self._on_data_cb(msg.get("payload", msg))

        elif event == "error" and self._on_error_cb:
            self._on_error_cb(Exception(msg.get("message", "Unknown server error")))