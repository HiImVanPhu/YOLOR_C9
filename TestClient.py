import socket

IP = "202.191.56.104"
PORT = 5518

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
while True:
    try:
        s.connect((IP, PORT))
        print('connect successfully')
        break
    except:
        print('connect failed')
        continue
# s.connect((IP, PORT))

file_name = 'main.zip'

s.send(f'{file_name}'.encode("utf-8"))

with open(file_name, 'rb') as f:
    data = f.read()
    s.sendall(data)
    print('sent')