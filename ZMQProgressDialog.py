__author__ = 'Abuenameh'

import os
import sys
import gtk
import gobject
import zmq

from speed import gprogress

from threading import Thread
from Queue import Queue, Empty

def enqueue_input(socket,queue):
    while True:
        message = socket.recv()
        queue.put(message)

def poll(queue, prog):
    try:
        obj = queue.get_nowait()
    except Empty:
        pass
    else:
        if(obj == Empty):
            gtk.main_quit()
        else:
            prog.next()
    return True


def start():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5556")
    q = Queue()
    t = Thread(target=enqueue_input, args=(socket,q,))
    t.daemon = True
    t.start()
    steps = int(q.get())
    prog = gprogress(range(steps), size=steps).__iter__()
    gobject.timeout_add(1, poll, q, prog)
    return False

if __name__ == '__main__':
    # print os.path.dirname(os.path.realpath(__file__))
    gtk.gdk.threads_init()
    gobject.timeout_add(1000, start)
    gtk.main()
