from flask import Flask, render_template, request
import sys
import time
import threading
from DBN import dbn
from DataPreparation import data_processing
import ctypes
import inspect
from matplotlib.pyplot import plotting
import os
import random


app = Flask(__name__)
global deepbelief
deepbelief = None
global t
t  = None
global stopped
stopped = False

global rint
rint = None

@app.route('/server/update_rbm_img')
def update_rbm_img():
    '''
    get images from rbm.
    '''
    global rint
    if rint == None:
        rint = random.randrange(10*10**6)
    
    path = 'static/img/'
    paths = os.listdir(path)
    for p in paths:
        if 'rbm.GIF' in p:
            return path+p+"?dummy="+str(rint) # Avoid caching
    return ""

@app.route('/server/update_dbn_img')
def update_dbn_img():
    '''
    get images from dbn.
    '''
    global rint
    if rint == None:
        rint = random.randrange(10*10**6)
    
    path = 'static/img/'
    paths = os.listdir(path)
    for p in paths:
        if 'dbn.png' in p:
            return path+p+"?dummy="+str(rint) # Avoid caching
    return ""


@app.route('/server/stop_thread')
def stop_thread():
    '''
    Stop the deep belief
    net creation.
    '''
    t.raiseExc(KeyboardInterrupt)
    global stopped
    stopped = True
    return 'Thread stopping.'

@app.route('/server/get_progress')
def get_progress():
    '''
    Get the progress of the deep belief
    net creation.
    '''
    if deepbelief == None:
        return ""
    
    progress = deepbelief.get_progress()
    return str(progress)+"%"


@app.route('/server/get_output')
def get_output():
    '''
    Get the output of the deep belief
    net creation.
    '''
    if deepbelief == None:
        return ""
    
    output_str = ""
    for elem in deepbelief.get_output():
        output_str+=elem+"\n"
    return output_str


@app.route('/',methods=['GET'])
def index():
    global stopped
    if request.method == 'GET' and not stopped:
        if not (request.args.get("param1") == None and request.args.get("param2") == None and request.args.get("param3") == None and request.args.get("param4") == None):
            global deepbelief
            try:
                vis = int(request.args.get("param1"))
                prehid = request.args.get("param2")
                hid = [int(elem) for elem in prehid.split(",")]
                out = int(request.args.get("param3"))
                epoch = int(request.args.get("param4"))
                deepbelief = dbn.DBN(vis,data_processing.get_batch_list(),hid,out,epoch,plot = True)
                remove_imgs()
                global t
                t = ThreadWithExc(target=deepbelief.run_dbn)
                t.start()
                return render_template('home.html',running = True, param1 = vis, param2 = prehid, param3 = out, param4 = epoch )
            except:
                return render_template('home.html',running = False,error = 'Please input parameters correct.')
        else:
            return render_template('home.html',running=False)
    else:
        stopped = False
        return render_template('home.html',running=False)

@app.route('/my-link/')
def my_link():
    print 'I got clicked!'

    return render_template('template.html',name = 'YES')


def remove_imgs():
    path = 'static/img/'
    paths = os.listdir(path)
    for p in paths:
        if 'rbm' in p:
            os.remove(path+p)
        elif 'dbn' in p:
            os.remove(path+p)
            
def _async_raise(tid, exctype):
    '''Raises an exception in the threads with id tid'''
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid,
                                                  ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")

class ThreadWithExc(threading.Thread):
    '''A thread class that supports raising exception in the thread from
       another thread.
    '''
    def _get_my_tid(self):
        """determines this (self's) thread id

        CAREFUL : this function is executed in the context of the caller
        thread, to get the identity of the thread represented by this
        instance.
        """
        if not self.isAlive():
            raise threading.ThreadError("the thread is not active")

        # do we have it cached?
        if hasattr(self, "_thread_id"):
            return self._thread_id

        # no, look for it in the _active dict
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._thread_id = tid
                return tid

        # TODO: in python 2.6, there's a simpler way to do : self.ident

        raise AssertionError("could not determine the thread's id")

    def raiseExc(self, exctype):
        """Raises the given exception type in the context of this thread.

        If the thread is busy in a system call (time.sleep(),
        socket.accept(), ...), the exception is simply ignored.

        If you are sure that your exception should terminate the thread,
        one way to ensure that it works is:

            t = ThreadWithExc( ... )
            ...
            t.raiseExc( SomeException )
            while t.isAlive():
                time.sleep( 0.1 )
                t.raiseExc( SomeException )

        If the exception is to be caught by the thread, you need a way to
        check that your thread has caught it.

        CAREFUL : this function is executed in the context of the
        caller thread, to raise an excpetion in the context of the
        thread represented by this instance.
        """
        _async_raise( self._get_my_tid(), exctype )


if __name__ == '__main__':
    app.run(debug=True)