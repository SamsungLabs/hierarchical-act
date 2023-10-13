import os
import shutil
import sys
from absl import app, flags

import gin
from core import Task, Serving

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_dir', None, 'Path of exp dir')
flags.DEFINE_string('data_root', '/home/sr5/dlvr/dataset/RT', 'Data root dir will be concatenated with data path in data_config.gin')
flags.DEFINE_string('data_config', None, 'Gin config for dataset')
flags.DEFINE_string('model_config', None, 'Gin config for model')
flags.DEFINE_string('task_config', None, 'Gin config for task')
flags.DEFINE_multi_string('gin_param', None, 'List of Gin parameter bindings')
flags.DEFINE_integer('gpu', 0, 'Path of exp dir')

def main(argv):
    exp_dir = FLAGS.exp_dir
    data_root = FLAGS.data_root
    data_config = FLAGS.data_config
    model_config = FLAGS.model_config
    task_config = FLAGS.task_config
    gpu = FLAGS.gpu
    
    if data_config is None:
        data_config = os.path.join(exp_dir, 'data.gin')
    if model_config is None:
        model_config = os.path.join(exp_dir, 'model.gin')
    if task_config is None:
        task_config = os.path.join(exp_dir, 'task.gin')
    
    gin_files = [data_config, model_config, task_config]
    gin_params = FLAGS.gin_param
    gin.parse_config_files_and_bindings(gin_files, gin_params)

    task = Task(exp_dir=exp_dir,
                data_root=data_root)
    serving = Serving(task, gpu=gpu)
    serving.serve()

if __name__ == '__main__':
    app.run(main)

