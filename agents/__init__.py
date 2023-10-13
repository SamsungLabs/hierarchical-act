from agents import hact_vq
from agents import multiact

def make(model_id, env_id):
    agent_module = globals()[model_id]
    agent_make_func = getattr(agent_module, "make")
    return agent_make_func(env_id)    
