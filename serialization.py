import marshal as m
import cPickle as p

def load(file):
    try:
        return m.load(file)
    except:
        path = file.name
        file.close() # Make sure that the file is properly closed.
        return p.load(open(path,'rb'))

def dump(value,file):
    m.dump(value,file)