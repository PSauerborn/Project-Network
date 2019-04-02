import cgi
import time



def notfound_404(environ, start_response):
    """Function that is called by default when the request made does not match any registered function"""
    start_response('404 Not Found', [('Content-type', 'text/plain')])

    return [b'Not Found']

class PathDispatcher():
    """Application that implements WSGI specifications to handle requests"""

    def __init__(self, dispatch_addr):

        # the pathmap dictionary contains the registered functions that are called when a request is made. Note that the keys are tuples of form (request_method, path)

        IP, PORT = dispatch_addr

        print('WSGI Interface Live.... Waiting for HTTP requests at IP: {} PORT: {}'.format(IP, PORT))

        self.pathmap = {}



    def __call__(self, environ, start_response):
        """ Method that is called when a request is made. Note that the __call__ function must be implemenetd for the application to work

        Parameters
        ----------
        environ: dict
            dictionary containing CGI like variables detailing the parameters of any request made
        start_response: function object
            callback function that is used by the application itself to send HTTP rquests and status Codes to the underlying server

        Returns
        -------

        handler: function object
            function that returns the actual data requested. Note the calling signature function(environ, start_response)


        """

        # the path/resource requested is first extracted. Note that the path along with the request method form the keys that determine which function is called to handle the request

        path = environ['PATH_INFO']

        # query parameters are extraced using the FieldStorage function from the cgi module. The FieldStorage function stores the values in a dictionary for later use

        params = cgi.FieldStorage(environ['wsgi.input'], environ=environ)

        # the request method is then extracted. Note that request method forms part of the key used for retrieving the correct handle method

        method = environ['REQUEST_METHOD'].lower()

        # parameters extracted using the cgi.FieldStorage function are then stored in the original environ dictionary in a slightly different format. This can be done at function level as well, if desired

        environ['params'] = {key: params.getvalue(key) for key in params}

        # the specified handeling method is then callled. Note that the get() function is used and, if the request tries to call a handler function not specified, the notfound_404 function defined above is returned instead

        handler = self.pathmap.get((method, path), notfound_404)

        # the handler function

        return handler(environ, start_response)

    def register(self, method, path, function):
        """Method used to register functions

        Parameters
        ----------
        environ: dicitonary
            dictionary containing values provided by web server
        method: str
            request method i.e. GET, POST etc
        path: str
            url path
        function: function object
            function to handle request

        """

        self.pathmap[method.lower(), path] = function
        return function
