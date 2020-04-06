from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi
import socket, time
from threading import Thread
import webbrowser
from PIL import Image
import io

"""Hosting the web server on here and Pi communication from HTTPServer
    One button to start image recording and show live feed on web page
    Button to classify current frame
    Return captured frame with classification info
"""

VERBOSE = False
IP_ADDRESS = "192.168.1.40"
IP_PORT = 22001

sock = None
isRunning = True

def debug(text):
    if VERBOSE:
        print("Debug:---", text)

# --------------------- class Receiver ---------------------------
class Receiver(Thread):
    def run(self):
        debug("Receiver thread started")
        while True:
            try:
                rxData = self.readServerData()
            except:
                debug("Exception in Receiver.run()")
                isReceiverRunning = False
                closeConnection()
                break
            debug("Receiver thread terminated")

    def readServerData(self):
        global isJPEG
        debug("Calling readResponse")
        bufSize = 4096
        data = bytearray(b"")
        while bytes(data)[-2:] != b"\xff\xd9":
        # eof tag for jpeg files (both chars must be in same block)
        # We are not sure 100% that this sequence is never embedded in image
        # but it is improbable to happen at the end of the data block
            try:
                blk = sock.recv(bufSize)
                if blk != None:
                    debug("Received data block, len: " + str(len(blk)))
                else:
                    debug("sock.recv() returned with None")
                data += blk
            except:
                raise Exception("Exception from blocking sock.recv()")
        print("JPEG received. Displaying it...")
        display(bytes(data))
# -------------------- End of Receiver ---------------------

def startReceiver():
    debug("Starting Receiver thread")
    receiver = Receiver()
    receiver.start()

def sendCommand(cmd):
    # debug("sendCommand() with cmd = " + cmd)
    try:
        sock.sendall(cmd)
    except:
        debug("Exception in sendCommand()")
        closeConnection()

def closeConnection():
    debug("Closing socket")
    sock.close()

def connect():
    global sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    debug("Connecting...")
    try:
        sock.connect((IP_ADDRESS, IP_PORT))
    except:
        debug("Connection failed.")
        return False
    startReceiver()
    return True

def display(data):
    jpgFile = "/Users/georgesnomicos/fridge_net/test.jpg"
    stream = io.BytesIO(data)
    image = Image.open(stream)
    print(image.size)
    image.save(jpgFile)

def saveData(data, filename):
    file = open(filename, "wb")
    file.write(data)
    file.close()

def onExit():
    global isRunning
    isRunning = False
    dispose()

class pageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('/'):

            type = 'text/html'

            self.send_response(200)
            self.send_header('content_type',type)
            self.end_headers()

            with open('text.html') as output:
                self.wfile.write(output.read().encode())

        if self.path.endswith('img'):
        #     cnt = 0
        #     if isRunning:
        #         # if cnt == 0:
        #         print("Sending command 'go'...")
        #         sendCommand(b"go")
        #         # else:
        #         #     print("Sending command ''...")
        #         #     sendCommand("")
        #         # Change sendCommand or write a condition from a push button
        #         # Test what happens to communication if sending ""
        #         time.sleep(5)
        #         cnt += 1

            type = 'image/jpg'
            with open('test.jpg','rb') as output:
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header('content_type',type)
                self.end_headers()
                # Link to raspberry pi here in another thread
                self.wfile.write(output.read())

    def do_POST(self):
        # Another thread analyse the image upon button press and display predicted class

        if self.path.endswith('/'):
            ctype, pdict = cgi.parse_header(self.headers.get('Content-Type'))
            pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
            pdict['CONTENT-LENGTH'] = int(self.headers.get('Content-length'))
            if ctype == 'multipart/form-data':
                fields = cgi.parse_multipart(self.rfile, pdict)
                print(fields)
                button_state = fields.get('button')
                print(button_state)
                send_command(button_state[0])

            self.send_response(301)
            self.send_header('content_type','text/html')
            self.send_header('Location', '/')
            self.end_headers()


def send_command(cmd):
    if cmd == "Forward":
        print('none')

if __name__ == "__main__":
    # if connect():
    #     print("Connection established")
    # else:
    #     print("Connection to %s:%d failed" %(IP_ADDRESS, IP_PORT))
    # time.sleep(1)

    PORT = 8080
    server_address = ("", PORT)
    server = HTTPServer(server_address, pageHandler)
    print(f"Server running on port {PORT}")
    server.serve_forever()
