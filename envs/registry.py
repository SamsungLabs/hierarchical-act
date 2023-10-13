env= {}

def make(id):
    cls_ = env[id]
    return cls_

def register(id):
    def _register(cls_):
       env[id] = cls_ 
       return cls_
    return _register