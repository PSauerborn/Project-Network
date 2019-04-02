from socket import *
from threading import Thread
import json
from Gutzwiller.gutzwiller_functions import *
from Gutzwiller.main import plot_data
import pickle
from multiprocessing import Process, Pool

class RPCHandler():
    """Class that handles the RPC calls. The class requires functions to be registered with it, which is stores in a dictionary"""

    def __init__(self):
        self._functions = {}

    def register_function(self, func):
        """Method to register function with Handler

        Parameters
        ----------
        func: function object
            function to be registered

        """
        self._functions[func.__name__] = func

    def __call__(self, connection):
        """Method called when the server the handler class belongs to handles a request

        connection: RPCProxy class
            socket object wrapped in proxy class

        """

        try:
            while True:

                # the function name and arguments are send by the connection as a pickled tuple

                func_name, args, kwargs = pickle.loads(connection.recv(1028))

                try:

                    # the requested function is then called and evaluated, and the results are pickled and sent back to the connection

                    result = self._functions[func_name](*args, **kwargs)

                    connection.send(pickle.dumps(result))

                except Exception as err:
                    print('Error Thrown in RPC Handler', err)

                    connection.send(pickle.dumps(err))

        except EOFError:
            pass


class Server(socket):
    """Sub-server that handles RPC call through the RPCHandler class. The Server class reqisters with the AdminServer upon instantiation, and a wrapped socket is then created to allow communication between the
    Server class and the AdminServer.

    Parameters
    ----------
    admir_addr: tuple
        tuple of form (HOST, PORT) specifying address of AdminServer
    addr: tuple
        tuple of form (HOST, PORT) specifying address of Server
    bufsiz: int
        buffer size

    """

    def __init__(self, admin_addr, addr=('localhost', 10024), bufsiz=1028):

        self.addr = (gethostbyname(gethostname()), 10024)

        super().__init__(AF_INET, SOCK_STREAM)

        # an RPCHandler object is created. Note that any function calls that are not found in the Server Class are delegated to the RPCHandler class (see __getattr__ method)

        self._handler = RPCHandler()

        # the server then registers itself with the admin server

        self.register_server(admin_addr)

    def __getattr__(self, name):
        return getattr(self._handler, name)

    def serve_forever(self):
        """Method to bind server to address and have it wait for response. Note that technically, the server gets only one single request in its lifetime. Once the proxy socket is connected to the server
        (and more importantly to its RPCHandler class), all the RPC's are handled by the RPCHandler class, not by the server itself.

        """

        self.bind(self.addr)
        self.listen()

        while True:

            print('Waiting for Connection at IP: {} PORT: {}'.format(gethostbyname(gethostname()), self.addr[1]))

            client_socket, client_addr = self.accept()

            print('Made Connection at {}'.format(client_addr))

            # once the server has been registered with the AdminServer, the AdminServer creates a wrapped socket and connects it to the Server it just registered.

            t = Thread(target=self._handler, args=(client_socket, ))
            t.daemon = True
            t.start()

    def register_server(self, serv_addr):
        """Method that registers a function with the AdminServer

        Parameters
        ----------
        serv_addr: tuple
            tuple of form (HOST, PORT) specifiying the address of Admin Server

        """

        print('Registering with Administator Server....')

        # the sub-server connects to the the Admin Server and sends a dictionary containing the desired service (in this case to register) and its address

        sock = socket(AF_INET, SOCK_STREAM)
        sock.connect(serv_addr)

        # the data is sent in the format of a pickled dictionary

        sock.send(pickle.dumps({'service': 'register', 'addr': self.addr}))


def HelloWorld(name):
    return 'Hello World, my name is {}'.format(name)

def get_phase(params):
    """Function registerd with the RPCHandler

    Parameters
    ----------
    params: dict
        dictionary containing system details needed to solve the problem

    Returns
    -------
    data: list
        list containing data in form of arrays from each quadrant

    """

    nrows, ncols, yrange, V, N, z = int(params['nrows']), int(params['ncols']), params['yrange'], float(params['V']), int(params['N']), int(params['z'])
    server_count = params['server_count']

    data = solve_system(server_count, yrange=yrange, nrows=nrows, ncols=ncols, V=V, N=N, z=z, U=1, N_sites=9, vary_mu=True, inhomogenous=False)

    return data


def solve_system(server_count, yrange, N_sites, V, nrows=10, ncols=10, mu=1, z=6, N=4, inhomogenous=False, vary_mu=False, U=None, target=plot_phase, total_iterations=200, CG=False):
    """Function that uses multiprocessing unit to split parameter space into quadrants and solve quadrants in parallel, resulting is much superior calculation time

    Parameters
    ----------
    N_sites: int
        number of sites
    V: float
        nearest neighbour interaction strength
    nrows: int
        number of rows in quadrant split
    ncols: int
        number of columns in quadrant split
    mu: float
        chemical potential
    z: int
        coordination number
    N: int
        maximum occupation number
    inhomogenous: boolean
        solves inhomogenous system if set to True
    vary_mu: boolean
        varies mu if set to True; else, U is varied
    target: func object
        target function for thread; one of ['plot_phase', 'plot_n', 'plot_uncertainty']

    """

    # note that the x-axis domain is significantly shorter for the inhomgenous system as the coordiantion number is not used

    if not inhomogenous:
        quadrants = construct_quadrants(server_count, nrows=nrows, ncols=ncols, yrange=yrange, xrange=(0.01, 0.2))
    else:
        quadrants = construct_quadrants(server_count, rows=nrows, ncols=ncols, xrange=(0.01, 0.2), yrange=yrange)

    nquads = len(quadrants)

    iters = int(np.ceil(total_iterations / nrows))

    threads = []
    processes = []

    pool = Pool(processes=nquads)

    data = []

    for n in quadrants.keys():

        quad, range = quadrants[n]

        xrange, yrange = range

        if __name__ == '__main__':

            p = pool.apply_async(target, (server_count, iters, mu, z, N, N_sites, V, xrange, yrange, quad, inhomogenous, vary_mu, U, CG))
            processes.append(p)

    for p in processes:
        result = p.get()
        data.append(result)

    return data



if __name__ == "__main__":

    IP = gethostbyname(gethostname())

    serv = Server(admin_addr=(IP, 8080))

    # functions are then registered with the RPCHandler; note that this is delegated

    serv.register_function(HelloWorld)
    serv.register_function(get_phase)

    serv.serve_forever()
