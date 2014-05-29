__author__ = 'Abuenameh'

import os
import sys
import pickle
import gtk
import gobject

from speed import gprogress

from subprocess import PIPE, Popen
from threading import Thread
from Queue import Queue, Empty

try:
    import cPickle as pickle
except:
    import pickle

def enqueue_input(queue):
    while True:
        try:
            data = pickle.load(sys.stdin)
            queue.put(data)
        except Exception as e:
            print e.message
            queue.put(Empty)
            break

def poll(queue, prog):
    try:
        queue.get_nowait()
    except Empty:
        pass
    else:
        obj = queue.get()
        if(obj == Empty):
            gtk.main_quit()
        else:
            prog.next()
    return True


def start():
    q = Queue()
    t = Thread(target=enqueue_input, args=(q,))
    t.daemon = True
    t.start()
    steps = q.get()
    prog = gprogress(range(steps), size=steps).__iter__()
    prog.next()
    gobject.timeout_add(1, poll, q, prog)
    return False

if __name__ == '__main__':
    print os.path.dirname(os.path.realpath(__file__))
    gtk.gdk.threads_init()
    gobject.timeout_add(1000, start)
    gtk.main()
