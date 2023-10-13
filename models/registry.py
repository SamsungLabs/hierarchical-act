model= {}

def make(id):
    cls_ = model[id]
    return cls_

def register(id):
    def _register(cls_):
       model[id] = cls_ 
       return cls_
    return _register