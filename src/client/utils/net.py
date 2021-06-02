
import socket             


class ClientNet():
    def __init__(self, server_addr="127.0.0.1", server_port=None):
        if server_port == None:
            raise "Incalid server port."
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((server_addr, server_port))

    def recv(self, recv_len):
        if recv_len <= 10000:
            return self.sock.recv(recv_len)
        else:
            msg = "".encode()
            received_len = 0
            while received_len < recv_len:
                # print("received %d bytes of %d bytes" % (received_len, recv_len))
                msg  += self.sock.recv(10000)
                received_len += 10000

            return msg

    def send(self, data):
        return self.sock.send(data)

class Client():
    def __init__(self, config=None) -> None:
        self.net = config