import sys
from socket import *
from threading import Thread
import json
import pickle
import requests
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageTk

LARGE_FONT = ('Verdana', 12)


class Application(tk.Tk):
     """Defines the parent frame of the application; note that the following application works by stacking frames on top of each other in the self._frames dictionary
    The correct frame can then be called and brought up to the front when desired

    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # container object is first defined and configured

        container = tk.Frame(self)
        container.pack(fill="both", expand=True, side="top")

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self._frames = {}

        # the inidividual 'pages' (frames really) are then instantiated and stored in the dictionary

        for F in (ConnectionPage, MainFrame):

            frame = F(container, self)
            self._frames[F] = frame

            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(ConnectionPage)

    def show_frame(self, container):
        """Method used to bring frame to the front"""

        frame = self._frames[container]
        frame.tkraise()

    def StartConnection(self, addr):
        """Method called once the client has successfully connected to the server"""

        self._frames[MainFrame].server_addr = addr
        self.geometry('950x600')
        self.show_frame(MainFrame)


class ConnectionPage(tk.Frame):
    """First Frame shown that allows user to enter the Host name and Port of the server waiting for the HTTP request"""

    def __init__(self, parent, controller):

        super().__init__(parent)

        self.controller = controller

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # a several title label are defined

        title = tk.Label(self, text='Please Enter Address of Path Dispatcher', font=LARGE_FONT).grid(row=0, sticky='sew', columnspan=3)
        label1 = tk.Label(self, text='IP: ').grid(row=1, column=0, sticky='se', padx=20)
        label2 = tk.Label(self, text='Port: ').grid(row=2, column=0, sticky='ne', padx=20)

        # Input boxes are added for the user to add the server address

        self.ip = tk.Entry(self)
        self.ip.grid(row=1, column=1, sticky='sw', columnspan=2)

        self.port = tk.Entry(self)
        self.port.grid(row=2, column=1, sticky='nw', columnspan=2)

        # default values are inserted into said entry fields

        self.ip.insert(tk.END, '192.168.0.8')
        self.port.insert(tk.END, '20000')

        # a button to submit the information and a button to quit program are both added.

        start = tk.Button(self, text='Advance', command=lambda: self.submit())
        start.grid(row=3, column=0, sticky='ne')

        quit = tk.Button(self, text='Quit', command=lambda: sys.exit())
        quit.grid(row=3, column=1, sticky='nw')

    def submit(self):
        """Method called to sumbit server information and connect to server; the GUI tests whether the information entered is correct by making a
        a HelloWorld request to the HTTP server.

        """

        # PORT and IP numbers are extracted from text fields

        IP, PORT = self.ip.get(), self.port.get()
        points = IP.count('.')

        # if the IP address doesnt contain 3 '.' characters (i.e. 192.168.0.8) then it is automatically invalid and hence not checked

        if points != 3:

            messagebox.showerror("Error!", "IP Address is not in Correct Format. Please Enter A Valid IP Address")
            return None

        # a HelloWorld Request is made; appropriate error messages are thrown if the request is invalid

        try:
            response = requests.get("http://{}:{}/hello".format(IP, PORT))

            if not response:

                messagebox.showerror('Error', 'HTTP Error: Status Code {}'.format(response.status_code))
                return None

            else:
                print('Request Successful')

        except Exception as err:

            messagebox.showerror('Error', 'Something went Wrong: {}'.format(err))
            return None

        # if the request is successful, the GUI moves on to the MainFrame

        self.controller.StartConnection((IP, PORT))


class StatusBarWidget(tk.Frame):
    """Widget that displays the status of the server along with a button to check wheter or not server is still live.
    The check is done via a HelloWorld request

    """

    def __init__(self, parent):

        super().__init__(parent, relief=tk.RAISED, borderwidth=2)

        self.server_addr = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1)

        # the widget contains three basic parts; the first is a title label

        title = tk.Label(self, text="Server Status: ", font=('Verdana', 12))
        title.grid(row=0, column=0, sticky='nsew')

        # the second is a button to test the connection

        test_server = tk.Button(self, text='Test Connection', command=lambda: self.test_server())
        test_server.grid(row=0, column=2, sticky='nsew')

        # the third is a label that is green when the server is live and red otherwise

        self.srv_status = tk.Label(self, text='Server Live', background='limegreen')
        self.srv_status.grid(row=0, column=1, sticky='nsew')


    def test_server(self):
        """Method called when server is tested. Sends a HelloWorld request to the HTTP server to check
        whether or not the server is live

        """

        # IP and PORT numbers are retrieved

        IP, PORT = self.server_addr[0], self.server_addr[1]

        # request is made

        try:
            response = requests.get("http://{}:{}/hello".format(IP, PORT))

            # HTTP errors are handled

            if not response:

                messagebox.showerror('Error', 'HTTP Error: Status Code {}'.format(response.status_code))

                # background is set to red in case of error

                self.srv_status.config(background='red', text='Server Down')

                return None

            else:
                print('Request Successful')

        except Exception as err:

            # other errors are caught; background is set to red in case of error

            self.srv_status.config(background='red', text='Server Down')

            messagebox.showerror('Error', 'Something went wrong: {}'.format(err))

            return False

        # background is set to green if server is live

        self.srv_status.config(background='limegreen', text='Server Live')

        return True


class MainContentWidget(tk.Frame):
    """Widget that makes up the majority of the content on the GUI. The entire GUI frame is split into two widgets, the first is the ServerStatusBar which gives the status of the server and the second is the
    content widget

    """

    def __init__(self, parent):

        self.parent = parent

        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=10)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=5)

        # the content widget is split into three sub-frames; the data frame is used to display the data png

        self.data_frame = tk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
        self.data_frame.grid(row=0, column=1, rowspan=2, sticky='nsew')

        # the 'choice' frame displays the drop down menu for the selection of the data type

        choice = tk.Frame(self, relief=tk.RAISED, borderwidth=2)
        choice.grid(row=0, column=0, sticky='nsew')

        # the 'param_frame' lets the user set the parameters of the system

        param_frame = tk.Frame(self, relief=tk.RAISED, borderwidth=1)
        param_frame.grid(row=1, column=0, sticky='nsew')

        # the choice frame is configured

        choice.grid_rowconfigure(0, weight=1)
        choice.grid_columnconfigure(0, weight=1)
        choice.grid_columnconfigure(1, weight=1)

        # label is added

        label = tk.Label(choice, text='Choose Data Type: ', font='Verdana 10 bold')
        label.grid(row=0, column=0, sticky='e')

        # dropdown menu is added

        options = ['Phase', 'Number', 'Density']

        Variable = tk.StringVar(self)
        Variable.set('Phase')

        menu = tk.OptionMenu(choice, Variable, *options, command=lambda x: self.handle(Variable.get()))
        menu.grid(row=0, column=1, sticky='w')

        self.state = 'phase'

        # the parameter frame is then configured

        param_frame.grid_columnconfigure(0, weight=1)
        param_frame.grid_columnconfigure(1, weight=1)

        for i in range(4):
            param_frame.grid_rowconfigure(i, weight=1)

        param_frame.grid_rowconfigure(4, weight=5)

        # labels and text fields are added to the widget to enter the corresponding parameters

        label = tk.Label(param_frame, text="Set Parameter Values", font='Verdana 10 bold')
        label.grid(row=0, column=0, sticky='n', pady=20, columnspan=2)

        params = ['Nmax', 'z', 'V']

        self.fields = []

        for i, x in enumerate(params, start=1):
            l = tk.Label(param_frame, text=x + ': ', font=('Verdana', 10))
            l.grid(row=i, column=0, sticky='e')

            e = tk.Entry(param_frame)
            e.grid(row=i, column=1, sticky='w')

            self.fields.append(e)

        # button is then added to send the request and get the data

        button = tk.Button(param_frame, text="Get Data", command=lambda: self.get_data())
        button.grid(row=4,column=0, columnspan=2)


    def handle(self, val):
        self.state = val.lower()

    def get_data(self):
        """Method that makes request to HTTP server and actives RCP on server side"""

        # entered parameters are extracted

        fields = [field.get() for field in self.fields]

        server_addr = self.parent.server_addr

        if server_addr is not None:

            # request string is built

            url = 'http://{}:{}/{}?N={}&z={}&V={}'.format(server_addr[0], server_addr[1], self.state, *fields)

            try:
                response = requests.get(url)

                if not response:

                    messagebox.showerror('Error', 'HTTP Error: Status Code {}'.format(response.status_code))

                    return None

                else:
                    print('Request Successful')

            except Exception as err:

                messagebox.showerror('Error', 'Something went wrong: {}'.format(err))

                return False

            # the server returns a PNG; this is saved and then read back and displayed to the widget

            data = response.content

            with open(r'./test.png', 'wb') as f:
                f.write(data)

            load = Image.open(r'./test.png')
            render = ImageTk.PhotoImage(load)


            img = tk.Label(self.data_frame, image=render)
            img.image = render
            img.pack()





class MainFrame(tk.Frame):
    """The main content Widget; consists of two smaller widgets; the first displays a server
    status bar (see above) and the second displays the data and text fields where parameters
    are set and request are made

    """

    def __init__(self, parent, controller):

        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=5)

        self.grid_columnconfigure(0, weight=1)

        self.status_bar = StatusBarWidget(self)
        self.status_bar.grid(row=0, column=0, sticky='nsew', pady=10)
        self.status_bar.grid_propagate(False)

        contentWidget = MainContentWidget(self)
        contentWidget.grid(row=1, column=0, sticky='nsew')
        contentWidget.grid_propagate(False)

        self.server_addr = None

    def __setattr__(self, attr, val):

        if attr == "server_addr":
            self.status_bar.server_addr = val
            return super().__setattr__(attr, val)

        return super().__setattr__(attr, val)





if __name__ == "__main__":
    
    app = Application()
    app.geometry('400x200')
    app.mainloop()
