from socketserver import TCPServer, BaseRequestHandler
from socket import *
from PathDispatcher import *
from wsgiref.simple_server import make_server
from threading import Thread
import json
import pickle
from time import sleep
from Gutzwiller.main import plot_data

class DataThread(Thread):
    """Object used to store data from Threaded process. Creating a container for the Thread allows data to be saved effectively

    Parameters
    ----------
    connection: proxied socket object
        socket object linked to the RPC serber that has been wrapped with the proxy class
    params: dict
        dictionary containing parameters

    """

    def __init__(self, connection, params):

        self.connection = connection
        self.params = params

        super().__init__()

    def run(self):
        """The run method is called when the thread is started. The thread calls the .get_phase() method, which the proxied connection passes on to the RPCHandler class, which in turn passes the request to the underlying TCP
        server. The TCP server then triggers the remote procedure call, and the data is stored in the Thread object

        """

        self.data = self.connection.get_phase(self.params)



class RPCProxy():
    """Class that acts as a proxy for a socket and exeutes the remote procedure call. Note that the function called must be registerd with the server that the connection is made with

    Parameters
    ----------
    connection: socket object
        socket connected to the chosen server. Note that the connection should already have been made
    server_addr: tuple
        tuple of form (HOST, PORT) specifiying address of the server the socket is connected to


    """

    def __init__(self, connection, server_addr):

        self._connection = connection
        self.server_addr = server_addr

    def __getattr__(self, name):

        def do_rpc(*args, **kwargs):
            """Closure that handles the actual RPC; the function name and arguments are packed into a tuple and then serialized using pickle. The byte stream is then sent to the server over a TCP connection
            and if the function called is registerd with the server, the server will execute the function and return the result

            """

            try:
                self._connection.send(pickle.dumps((name, args, kwargs)))

                result = pickle.loads(self._connection.recv(400000))

                if isinstance(result, Exception):
                    raise result

            except Exception as err:
                print('Error Thrown in Proxy', err)

            return result
        return do_rpc

    def __str__(self):
        return 'Bound to IP: {} PORT: {}'.format(self.server_addr[0], self.server_addr[1])


class AdminServer(socket):
    """Admin server that keeps a record of active sub-servers and then handle the RPC. The Admin Server is linked to the subservers via a wrapped socket. All methods registered with the sub-server can be accessed with is proxy.
    The AdminServer receives and interprets HTTP request with its PathDispatcher object. It then does an RPC to the sub servers registered with it, which obtain the data and send it back to the Admin Server, which then returns
    it to the client.

    Parameters
    ----------
    addr: tuple
        tuple of form (HOST, PORT) specifying the address of the server
    dispatcher: PathDispatcher object
        WSGI interface object that handles incoming HTTP requests

    """

    _active_servers = []

    def __init__(self, addr, dispatcher=None):

        super().__init__(AF_INET, SOCK_STREAM)

        self.addr = addr

        IP = gethostbyname(gethostname())

        # the path dispatcher object is instatiated, and the desired functions are registerd with the dispatcher

        dispatcher = PathDispatcher((IP, 20000))
        dispatcher.register('GET', '/phase', self.get_phase)
        dispatcher.register('GET', '/hello', self.HelloWord)

        # the server is then set to listen for requests

        serv = make_server('', 20000, dispatcher)

        t = Thread(target=serv.serve_forever)
        t.start()


    def serve_forever(self):
        """Method that activates the server"""

        self.bind(self.addr)
        self.listen()

        IP = gethostbyname(gethostname())

        while True:

            print('Waiting for connections at IP: {} PORT: {}'.format(IP, self.addr[1]))

            client_socket, client_addr = self.accept()

            print('Connection Made at {}'.format(client_addr))

            # the client socket is passed down to the .handle() method

            t = Thread(target=self.handle, args=(client_socket, ))
            t.start()

    def register_server(self, serv_addr):
        """Function that registers a sub-server with the Admin server

        Parameters
        ----------
        serv_addr: tuple
            tuple of form (HOST, PORT) specifiying the address of the sub-server being registered

        """

        # a new connection is made to the server

        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect(serv_addr)

        # the connection is then wrapped with the proxy class

        p = RPCProxy(sock, serv_addr)

        print('Registering Server at {}'.format(serv_addr))

        # the proxy is added to the list of active servers

        self._active_servers.append(p)

        print('Current Active Servers')

        for prox in self._active_servers:
            print(prox)


    def handle(self, client_socket):
        """Method that handles TCP connection. Note that the only direct connections the server receives are from the sub-servers running on the network when they register.
        All other requests are handled through the PathDispatcher object

        Parameters
        ----------
        client_socket: socket object
            client socket object

        """

        payload = pickle.loads(client_socket.recv(1028))

        if payload['service'] == 'register':
            self.register_server(payload['addr'])

        client_socket.close()

    def get_phase(self, environ, start_response):
        """Method called when a GET request is made with resource path /phase to the PathDispatcher

        Parameters
        ----------
        environ: dict
            dictionary containing information about the query
        start_response: func
            function object used to start a response

        """

        # a response is first started

        start_response('200 OK', [('Content-type', 'image/png')])

        params = environ['params']
        params['ncols'], params['nrows'] = 5, 5

        server_count = len(self._active_servers)
        incr = 3 / server_count

        # each sub-server then solves for a particular region of parameter space

        threads = []

        # an RPC is made to each registered sub-servers through their respective sockets

        for i, proxy in enumerate(self._active_servers):

            print('sending to...', proxy)

            # each sub-server solves for a different row of the phase diagrams

            params['yrange'] = (incr*i, incr*(i+1))
            params['server_count'] = i

            # the DataThread objects (which inherit from the Thread objects) are used so that data can be stored between threads

            t = DataThread(proxy, params)
            threads.append(t)
            t.start()

        # all threads are then joined; note that the sub-servers wil be unavaiable until all threads are finished

        for t in threads:
            t.join()

        print('Threads done')

        # the raw data (stored in the DataThread objects) is then passed down to the plot_function, which combines all the quadrants and plots the result

        fig = plot_data(server_count, threads, nrows=5, ncols=5, iters=40)

        # the figure is then saved, and loaded. This is neccesary because of the format that the images are in when generated by matplotlib. By saving and reloading, the images are converted into simply PNG's

        fig.savefig(r'C:\directory_python\PythonCookBook\NetworkProgramming\ProjectNetwork\figures\test.png')

        with open(r'C:\directory_python\PythonCookBook\NetworkProgramming\ProjectNetwork\figures\test.png', 'rb') as f:
            fig = f.read()

        # the results are then sent back to the PathDispatcher, which returns the result to the client

        return [fig]

    def HelloWord(self, environ, start_response):
        """Tester functions used to check wheter of not the server is working"""

        start_response('200 OK', [('Content-type', 'text/html')])

        for proxy in self._active_servers:
            resp = proxy.HelloWorld('pascal').encode('utf-8')

        return [resp]



if __name__ == "__main__":

    IP = gethostbyname(gethostname())

    a = AdminServer(addr=(IP, 8080))
    a.serve_forever()
