agent= {}

def make(id):
    cls_ = agent[id]
    return cls_

def register(id):
    def _register(cls_):
       agent[id] = cls_ 
       return cls_
    return _register