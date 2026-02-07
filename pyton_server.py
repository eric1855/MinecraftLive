import asyncio
import json
import ssl
import os
import sys
import socket
import threading
import http.server
import socketserver

# --- DEPENDENCY CHECK ---
try:
    import websockets
    from pynput.mouse import Button, Controller
except ImportError:
    print("Error: Missing dependencies.")
    print("Run: pip install pynput websockets")
    sys.exit(1)

# --- CONFIGURATION ---
HTTP_PORT = 8000
WS_PORT = 8765
SENSITIVITY_X = 6.0
SENSITIVITY_Y = 6.0
SMOOTHING_FACTOR = 0.3

mouse = Controller()

# --- HTML CLIENT ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Gyro Controller</title>
    <style>
        body { 
            background: #121212; margin: 0; overflow: hidden; 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            user-select: none; -webkit-user-select: none;
            display: grid; grid-template-rows: 1fr 120px; height: 100vh;
        }

        /* GYRO AIM AREA */
        #aim-zone {
            background: #222; color: #555;
            display: flex; flex-direction: column;
            justify-content: center; align-items: center;
            border-bottom: 2px solid #333;
            font-size: 1.2rem; font-weight: bold; text-transform: uppercase;
            position: relative;
        }
        #aim-zone.active { background: #2a2a2a; color: #0f0; }

        /* BUTTON ROW */
        #controls {
            display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px;
            background: #000;
        }

        button {
            border: none; border-radius: 12px;
            font-size: 1.5rem; font-weight: bold; color: white;
            touch-action: none;
        }

        #btn-left { background: #d32f2f; } /* Red */
        #btn-left:active { background: #b71c1c; }

        #btn-right { background: #1976d2; } /* Blue */
        #btn-right:active { background: #0d47a1; }
        
        /* STATUS & FIX LINK */
        #status-bar {
            position: absolute; top: 20px; 
            background: rgba(0,0,0,0.8); padding: 5px 15px; border-radius: 20px;
            font-size: 0.9rem; color: #aaa;
            z-index: 50;
        }
        
        #fix-link {
            margin-top: 15px;
            color: #ff3b30; text-decoration: underline; font-size: 1rem;
            cursor: pointer; display: none; z-index: 60;
        }

        /* OVERLAY */
        #overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.9); z-index: 100;
            display: flex; justify-content: center; align-items: center;
        }
        #start-btn { padding: 20px 40px; font-size: 1.2rem; background: #0f0; color: #000; border-radius: 50px; }
    </style>
</head>
<body>
    <div id="aim-zone">
        <div id="status-bar">Connecting...</div>
        <div id="aim-text">Hold Here to Look</div>
        <a id="fix-link" target="_blank">🔴 Tap here to Fix Trust</a>
    </div>

    <div id="controls">
        <button id="btn-left">ATTACK</button>
        <button id="btn-right">BUILD</button>
    </div>

    <div id="overlay"><button id="start-btn">TAP TO START</button></div>

    <script>
        // Inject Python variables
        const WS_URL = "wss://{{HOST_IP}}:{{WS_PORT}}";
        const FIX_URL = "https://{{HOST_IP}}:{{WS_PORT}}";

        const statusDiv = document.getElementById('status-bar');
        const fixLink = document.getElementById('fix-link');
        const aimText = document.getElementById('aim-text');
        
        let socket = null;
        let aimActive = false;

        // Set the fix link URL
        fixLink.href = FIX_URL;

        function connect() {
            statusDiv.innerText = "Connecting to Mouse...";
            
            socket = new WebSocket(WS_URL);
            
            socket.onopen = () => {
                statusDiv.innerText = "🟢 Connected";
                statusDiv.style.color = "#4cd964";
                fixLink.style.display = 'none';
                aimText.style.display = 'block';
            };

            socket.onclose = () => {
                statusDiv.innerText = "🔴 Disconnected";
                statusDiv.style.color = "#ff3b30";
                // Show the fix link if we can't connect
                fixLink.style.display = 'block';
                aimText.style.display = 'none';
            };

            socket.onerror = (err) => {
                console.error(err);
            };
        }

        function send(type, payload) {
            if (socket && socket.readyState === 1) {
                socket.send(JSON.stringify({ t: type, d: payload }));
            }
        }

        // --- SENSORS ---
        window.addEventListener('deviceorientation', (e) => {
            if (aimActive) send('move', { a: e.alpha, b: e.beta });
        });

        // --- TOUCH AREAS ---
        const aimZone = document.getElementById('aim-zone');
        aimZone.addEventListener('touchstart', (e) => { 
            // Don't trigger aim if tapping the fix link
            if(e.target === fixLink) return;
            e.preventDefault(); aimActive = true; aimZone.classList.add('active'); 
        });
        aimZone.addEventListener('touchend', (e) => { e.preventDefault(); aimActive = false; aimZone.classList.remove('active'); });

        // --- BUTTONS ---
        const btnLeft = document.getElementById('btn-left');
        const btnRight = document.getElementById('btn-right');

        btnLeft.addEventListener('touchstart', (e) => { e.preventDefault(); send('click', { b: 'left', s: 'down' }); });
        btnLeft.addEventListener('touchend', (e) => { e.preventDefault(); send('click', { b: 'left', s: 'up' }); });

        btnRight.addEventListener('touchstart', (e) => { e.preventDefault(); send('click', { b: 'right', s: 'down' }); });
        btnRight.addEventListener('touchend', (e) => { e.preventDefault(); send('click', { b: 'right', s: 'up' }); });

        // --- STARTUP ---
        document.getElementById('start-btn').addEventListener('click', () => {
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                DeviceOrientationEvent.requestPermission().then(r => {
                    if (r=='granted') {
                        document.getElementById('overlay').style.display = 'none';
                        connect();
                    }
                });
            } else {
                document.getElementById('overlay').style.display = 'none';
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

# --- HTTP SERVER (Threaded) ---
class ClientHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
            ip = get_local_ip()
            html = HTML_TEMPLATE.replace("{{HOST_IP}}", ip).replace("{{WS_PORT}}", str(WS_PORT))
            self.wfile.write(html.encode('utf-8'))
        else: self.send_error(404)
    def log_message(self, format, *args): pass

def start_http():
    ip = get_local_ip()
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile='cert.pem')
    
    server = socketserver.TCPServer(("", HTTP_PORT), ClientHandler)
    server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
    
    print(f"\n[+] CONTROLLER INTERFACE: https://{ip}:{HTTP_PORT}")
    server.serve_forever()

# --- WEBSOCKET SERVER (Async) ---
async def start_ws():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile='cert.pem')
    
    print(f"[+] MOUSE SERVER LISTENING: {WS_PORT}")
    
    # State variables
    prev_a, prev_b = None, None
    velocity_x, velocity_y = 0.0, 0.0

    async def handler(websocket):
        nonlocal prev_a, prev_b, velocity_x, velocity_y
        print("Phone Connected to Mouse Server!")
        try:
            async for message in websocket:
                msg = json.loads(message)
                m_type = msg.get('t')
                data = msg.get('d')

                if m_type == 'click':
                    btn = Button.left if data['b'] == 'left' else Button.right
                    if data['s'] == 'down': mouse.press(btn)
                    else: mouse.release(btn)

                elif m_type == 'move':
                    curr_a, curr_b = data['a'], data['b']
                    
                    if prev_a is not None:
                        da = curr_a - prev_a
                        db = curr_b - prev_b
                        
                        # Fix wrapping
                        if da > 180: da -= 360
                        if da < -180: da += 360

                        # Calculate Target
                        target_x = da * SENSITIVITY_X * -1
                        target_y = db * SENSITIVITY_Y * -1

                        # Smoothing (Lerp)
                        velocity_x = (velocity_x * (1 - SMOOTHING_FACTOR)) + (target_x * SMOOTHING_FACTOR)
                        velocity_y = (velocity_y * (1 - SMOOTHING_FACTOR)) + (target_y * SMOOTHING_FACTOR)

                        if abs(velocity_x) > 0.1 or abs(velocity_y) > 0.1:
                            mouse.move(velocity_x, velocity_y)
                            
                    prev_a, prev_b = curr_a, curr_b
        except Exception:
            pass
        finally:
            print("Phone Disconnected")

    # Use the serve method directly
    async with websockets.serve(handler, "0.0.0.0", WS_PORT, ssl=ssl_context):
        await asyncio.Future()

if __name__ == "__main__":
    if not os.path.exists("cert.pem"):
        print("Error: cert.pem not found.")
    else:
        # Start HTTP in background thread
        t = threading.Thread(target=start_http, daemon=True)
        t.start()
        
        # Start WebSocket in main thread
        try: asyncio.run(start_ws())
        except KeyboardInterrupt: print("\nStopped.")