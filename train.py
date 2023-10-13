import os
import shutil
import sys
from absl import app, flags

import gin
from core import Task, SingleGpuTrainer

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_dir', None, 'Path of exp dir')
flags.DEFINE_string('data_root', '/home/sr5/dlvr/dataset/RT', 'Data root dir will be concatenated with data path in data_config.gin')
flags.DEFINE_string('data_config', './configs/dataset/mujoco_bimanipulation/all_tasks.gin', 'Gin config for dataset')
flags.DEFINE_string('model_config', './configs/model/act/bimanipulation.gin', 'Gin config for model')
flags.DEFINE_string('task_config', './configs/task/act_bimanipulation_task.gin', 'Gin config for task')
flags.DEFINE_multi_string('gin_param', None, 'List of Gin parameter bindings')
flags.DEFINE_integer('gpu', 0, 'Path of exp dir')

def main(argv):
    exp_dir = FLAGS.exp_dir
    data_root = FLAGS.data_root
    data_config = FLAGS.data_config
    model_config = FLAGS.model_config
    task_config = FLAGS.task_config
    gpu = FLAGS.gpu
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    shutil.copyfile(data_config, os.path.join(exp_dir, 'data.gin'))
    shutil.copyfile(model_config, os.path.join(exp_dir, 'model.gin'))
    shutil.copyfile(task_config, os.path.join(exp_dir, 'task.gin'))

    gin_files = [data_config, model_config, task_config]
    gin_params = FLAGS.gin_param
    gin.parse_config_files_and_bindings(gin_files, gin_params)

    task = Task(exp_dir=exp_dir,
                data_root=data_root)
    trainer = SingleGpuTrainer(task, gpu=gpu)
    trainer.train()

if __name__ == '__main__':
    app.run(main)


