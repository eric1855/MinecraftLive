import asyncio
import json
import socket
import threading
import http.server
import socketserver
import ssl
import os
import sys

# Try to import required libraries
try:
    import websockets
    import pyautogui
except ImportError:
    print("Error: Missing dependencies. Run: pip install pyautogui websockets")
    sys.exit(1)

# --- CONFIGURATION ---
HTTP_PORT = 8000
WS_PORT = 8765
SENSITIVITY_X = 15
SENSITIVITY_Y = 15

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

# --- HTML CLIENT CODE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Gyro Mouse (Secure)</title>
    <style>
        body { 
            font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee; 
            text-align: center; display: flex; flex-direction: column; 
            justify-content: center; align-items: center; height: 100vh; margin: 0; 
        }
        #status { margin-bottom: 20px; color: #888; }
        #trackpad {
            width: 80vw; height: 50vh; background: #333; border-radius: 20px;
            display: flex; justify-content: center; align-items: center; border: 2px dashed #555;
        }
        .active-area { background: #007bff !important; }
        button { padding: 15px 30px; font-size: 18px; border-radius: 30px; background: #007bff; color: white; border: none; }
    </style>
</head>
<body>
    <h1>Gyro Mouse</h1>
    <div id="status">Ready</div>
    <button id="permBtn">Start Mouse</button>
    <div id="trackpad" style="display:none;"><span>Hold to Move</span></div>

    <script>
        const WS_URL = "wss://{{HOST_IP}}:"; // Note: wss:// for secure
        const WS_PORT = "{{WS_PORT}}";
        let socket = null;
        let isTouching = false;

        function connect() {
            socket = new WebSocket(WS_URL + WS_PORT);
            
            socket.onopen = () => { 
                document.getElementById('status').innerText = "Connected";
                document.getElementById('status').style.color = "#4cd964";
                document.getElementById('permBtn').style.display = 'none';
                document.getElementById('trackpad').style.display = 'flex';
            };
            
            socket.onerror = (err) => {
                document.getElementById('status').innerText = "Connection Failed. Trust the certificate!";
                console.error(err);
            };
        }

        function handleOrientation(event) {
            if (socket && socket.readyState === WebSocket.OPEN && isTouching) {
                socket.send(JSON.stringify({
                    alpha: event.alpha, beta: event.beta
                }));
            }
        }

        const trackpad = document.getElementById('trackpad');
        trackpad.addEventListener('touchstart', (e) => { e.preventDefault(); isTouching = true; trackpad.classList.add('active-area'); });
        trackpad.addEventListener('touchend', (e) => { e.preventDefault(); isTouching = false; trackpad.classList.remove('active-area'); });

        document.getElementById('permBtn').addEventListener('click', async () => {
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                try {
                    const response = await DeviceOrientationEvent.requestPermission();
                    if (response === 'granted') {
                        window.addEventListener('deviceorientation', handleOrientation);
                        connect();
                    } else { alert("Permission denied."); }
                } catch (e) { alert(e); }
            } else {
                window.addEventListener('deviceorientation', handleOrientation);
                connect();
            }
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
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            local_ip = get_local_ip()
            html = HTML_TEMPLATE.replace("{{HOST_IP}}", local_ip).replace("{{WS_PORT}}", str(WS_PORT))
            self.wfile.write(html.encode('utf-8'))
        else: self.send_error(404)
    def log_message(self, format, *args): pass

def start_servers():
    ip = get_local_ip()
    
    # 1. SETUP SSL CONTEXT
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile='cert.pem')

    print(f"\n[!] IMPORTANT: You are using a Self-Signed Certificate.")
    print(f"    On your phone, you will see a 'Not Private' warning.")
    print(f"    Click 'Show Details' -> 'visit this website' to proceed.\n")
    print(f"[1] Phone Link: https://{ip}:{HTTP_PORT}")

    # 2. START HTTP SERVER (SECURE)
    httpd = socketserver.TCPServer(("", HTTP_PORT), ClientHandler)
    httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # 3. START WEBSOCKET SERVER (SECURE)
    async def main():
        previous_data = {'alpha': None, 'beta': None}
        async def handler(websocket):
            nonlocal previous_data
            try:
                async for message in websocket:
                    data = json.loads(message)
                    curr_alpha = data['alpha']
                    curr_beta = data['beta']
                    
                    if previous_data['alpha'] is not None:
                        delta_alpha = curr_alpha - previous_data['alpha']
                        delta_beta = curr_beta - previous_data['beta']
                        if delta_alpha > 180: delta_alpha -= 360
                        if delta_alpha < -180: delta_alpha += 360
                        
                        move_x = delta_alpha * SENSITIVITY_X * -1
                        move_y = delta_beta * SENSITIVITY_Y * -1
                        if abs(move_x) > 0.5 or abs(move_y) > 0.5:
                            pyautogui.move(move_x, move_y)
                    previous_data = {'alpha': curr_alpha, 'beta': curr_beta}
            except: pass

        async with websockets.serve(handler, "0.0.0.0", WS_PORT, ssl=ssl_context):
            await asyncio.Future()

    asyncio.run(main())

if __name__ == "__main__":
    if not os.path.exists("cert.pem"):
        print("Error: cert.pem not found. Run the openssl command first!")
    else:
        try: start_servers()
        except KeyboardInterrupt: print("\nStopped.")