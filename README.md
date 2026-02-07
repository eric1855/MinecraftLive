Instruction for running the phone hand control:

ensure cert.pem is present

run pyton_server.py IN A DEDICATED WINDOW OTHERWISE ITS GOING TO STOP THE OTHER THING 

If it has errors in the directory of cert.pem run 

rm -rf cert.pem 

THEN:

openssl req -new -x509 -keyout cert.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Company/CN=localhost"

Then run the server

Open the server:

Go to the website on your phone. address is in the terminal

Go to the same one but change :8000 to :8765

Click visit anyways for both

You should see an error pop up in terminal

Navigate back to the :8000 one and refresh. It shoul connect after a bit.





todolist:
fix error in the pyinput thing that causes crashes

comment out old code like the mouse pointing code.


