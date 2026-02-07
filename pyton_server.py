import asyncio
import json
import socket
import threading
import http.server
import socketserver
import ssl
import os
import sys
import time

# --- INSTALL CHECK ---
try:
    from pynput.mouse import Controller
    import websockets
except ImportError:
    print("Error: Missing dependencies.")
    print("Run: pip install pynput websockets")
    sys.exit(1)

# --- CONFIGURATION (GAME TUNED) ---
HTTP_PORT = 8000
WS_PORT = 8765

# SENSITIVITY: Lower is often better for Minecraft to prevent spinning out
SENSITIVITY_X = 8.0  
SENSITIVITY_Y = 8.0 

# SMOOTHING (0.0 to 1.0): 
# 0.1 = Very smooth/slow (cinematic)
# 0.5 = Balanced
# 0.9 = Raw/Twitchy
SMOOTHING_FACTOR = 0.4 

mouse = Controller()

# --- HTML CLIENT (Same as before, simplified for space) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Gyro Gaming</title>
    <style>
        body { background: #121212; color: #0f0; font-family: monospace; text-align: center; overflow: hidden; }
        #area { width: 100vw; height: 100vh; display: flex; justify-content: center; align-items: center; touch-action: none; }
        .active { background: #1a1a1a; }
    </style>
</head>
<body>
    <div id="area">
        <div id="msg">Tap & Hold to Aim</div>
    </div>
    <script>
        const WS_URL = "wss://{{HOST_IP}}:";
        const WS_PORT = "{{WS_PORT}}";
        let socket = null;
        let isTouching = false;
        
        function connect() {
            socket = new WebSocket(WS_URL + WS_PORT);
            socket.onopen = () => document.getElementById('msg').innerText = "LINKED - HOLD TO AIM";
            socket.onclose = () => document.getElementById('msg').innerText = "DISCONNECTED";
        }

        function handle(e) {
            if (socket && socket.readyState === 1 && isTouching) {
                // Send Alpha (Yaw) and Beta (Pitch)
                socket.send(JSON.stringify({ a: e.alpha, b: e.beta }));
            }
        }

        const area = document.getElementById('area');
        area.addEventListener('touchstart', (e) => { 
            e.preventDefault(); isTouching = true; 
            area.classList.add('active'); 
            
            // Request permission on first tap
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                DeviceOrientationEvent.requestPermission().then(r => {
                    if (r=='granted') window.addEventListener('deviceorientation', handle);
                });
            } else {
                window.addEventListener('deviceorientation', handle);
            }
            if(!socket) connect();
        });
        
        area.addEventListener('touchend', (e) => { 
            e.preventDefault(); isTouching = false; 
            area.classList.remove('active'); 
        });
    </script>
</body>
</html>
"""

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(('8.8.8.8', 1)); IP = s.getsockname()[0]
    except: IP = '127.0.0.1'
    finally: s.close()
    return IP

class ClientHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
            self.wfile.write(HTML_TEMPLATE.replace("{{HOST_IP}}", get_local_ip()).replace("{{WS_PORT}}", str(WS_PORT)).encode('utf-8'))
    def log_message(self, format, *args): pass

def start_servers():
    ip = get_local_ip()
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile='cert.pem') # REUSES YOUR CERT FROM BEFORE

    print(f"\n[+] GAMING MODE ACTIVE")
    print(f"[+] Connect: https://{ip}:{HTTP_PORT}\n")

    # HTTP
    httpd = socketserver.TCPServer(("", HTTP_PORT), ClientHandler)
    httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # WEBSOCKET
    async def main():
        # SMOOTHING STATE
        prev_a, prev_b = None, None
        velocity_x, velocity_y = 0.0, 0.0
        
        async def handler(websocket):
            nonlocal prev_a, prev_b, velocity_x, velocity_y
            try:
                async for message in websocket:
                    data = json.loads(message)
                    curr_a, curr_b = data['a'], data['b']

                    if prev_a is not None:
                        # 1. Calculate Raw Delta
                        da = curr_a - prev_a
                        db = curr_b - prev_b
                        
                        # 2. Fix Wrapping (359 -> 0)
                        if da > 180: da -= 360
                        if da < -180: da += 360

                        # 3. Target Movement
                        target_x = da * SENSITIVITY_X * -1
                        target_y = db * SENSITIVITY_Y * -1

                        # 4. Apply Smoothing (Lerp)
                        # New velocity is a blend of old velocity and new target
                        velocity_x = (velocity_x * (1 - SMOOTHING_FACTOR)) + (target_x * SMOOTHING_FACTOR)
                        velocity_y = (velocity_y * (1 - SMOOTHING_FACTOR)) + (target_y * SMOOTHING_FACTOR)

                        # 5. Move Mouse (Send Relative Delta)
                        # We only move if the smoothed value is significant to prevent "drift" when still
                        if abs(velocity_x) > 0.1 or abs(velocity_y) > 0.1:
                            mouse.move(velocity_x, velocity_y)

                    prev_a, prev_b = curr_a, curr_b
            except Exception as e: pass

        async with websockets.serve(handler, "0.0.0.0", WS_PORT, ssl=ssl_context):
            await asyncio.Future()

    asyncio.run(main())

if __name__ == "__main__":
    if not os.path.exists("cert.pem"):
        print("Error: cert.pem missing. Run the openssl command!")
    else:
        try: start_servers()
        except KeyboardInterrupt: print("\nStopped.")