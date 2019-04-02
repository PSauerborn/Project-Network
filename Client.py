
from socket import *
from threading import Thread
import json
import pickle
import requests
import tkinter



class Application(tk.Tk):

    def __init__(self):

        super().__init__()

        container = tk.Frame(self)
