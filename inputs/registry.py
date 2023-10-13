datasets= {}

def make(id):
    cls_ = datasets[id]
    return cls_

def register(id):
    def _register(cls_):
       datasets[id] = cls_ 
       return cls_
    return _register