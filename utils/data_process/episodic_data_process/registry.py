processor= {}

def make(id):
    cls_ = processor[id]
    return cls_

def register(id):
    def _register(cls_):
       processor[id] = cls_ 
       return cls_
    return _register