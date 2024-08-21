import os
import torch
import multiprocessing as mp
from datetime import datetime
import sys
import traceback
from MIA.utils import ConfigClass
from helper import run_helper
import pathlib
from SERDatasets import IEMOCAPDatasetConstructor, PodcastDatasetConstructor, ImprovDatasetConstructor, MuSEDatasetConstructor

def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

# Workers have an expensive start up and shutdown cost
# so what we do is launch the maximum amount of workers across GPUs and wait for jobs
# results queue will be used to ensure that all jobs have been completed 
class WorkerManager:
    def __init__(self):
        self.num_scripts_per_gpu = 1 # How many jobs can be run simultaneously on a GPU
        self.num_gpus = torch.cuda.device_count()
        self.num_processes = self.num_gpus * self.num_scripts_per_gpu
        self.job_queue = mp.Queue()
        self.results_queue = mp.Queue()
        # self.available_gpus = mp.Queue()
        # Will add e.g. for 3 GPUs [0,0,0,1,1,1,2,2,2] -- subprocesses will be given items in this list assigning them a GPU
        self.available_gpus = [gpu for gpu in range(self.num_gpus) for _ in range(self.num_scripts_per_gpu)]
        self.processes = []

    def create_workers(self):
        for gpu in self.available_gpus:
            process = Worker(self.job_queue, self.results_queue, gpu)
            self.processes.append(process)
            process.start()

    def graceful_shutdown(self):
        # First clear the queue to prevent any unlaunched jobs from launching
        while not self.job_queue.empty():
            self.job_queue.get()
        # Passing 'shutdown worker' on the job queue will nicely shutdown a worker allowing it to finish it's current job
        # unless there is a crash this shutdown should always be used 
        for _ in range(self.num_processes):
            self.job_queue.put('shutdown worker')

    def launch_jobs(self, job_args, allow_crash=True):
        for script_name in job_args:
            # Tuple of script_name and the arguments for that script
            self.job_queue.put((script_name, allow_crash, job_args[script_name]))

    def wait_for_jobs(self, job_names):
        finished_jobs = []
        failed_jobs = []
        print('Waiting for jobs to finish:', job_names)
        while sorted(finished_jobs) != sorted(job_names):
            finished_job = self.results_queue.get(block=True, timeout=None)
            if type(finished_job) != str:
                print('Found a crashed job where allow_crash=False')
                script_name, exception_info = finished_job
                print(f'Crash information for {script_name}:')
                print(exception_info[0])
                print(exception_info[1])
                print('see script log for traceback information')
                print('Shutting down')
                self.graceful_shutdown()
                exit()
            else:
                if finished_job.startswith('BAD_'):
                    finished_job = finished_job.replace('BAD_', '')
                    failed_jobs.append(finished_job)
                finished_jobs.append(finished_job)

            print('Finished job(s):', finished_job, '(', finished_jobs, ')')
            print('Remaining jobs', set(job_names) - set(finished_jobs))
            print('Crashed jobs', failed_jobs)

class Worker(mp.Process):
    def __init__(self, job_queue, results_queue, gpu):
        super(Worker, self).__init__()
        self.job_queue = job_queue
        self.results_queue = results_queue
        self.gpu = gpu
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.gpu}'
        print(f'Worker initialised on GPU {self.gpu}')

    def run(self):
        print(f'Worker ({os.getpid()}) ready for jobs on GPU {self.gpu}')
        # Once shutdown worker is read on the job queue the worker will be killed
        for (script_name, allow_crash, args) in iter(self.job_queue.get, 'shutdown worker'):
            print(f'Worker ({os.getpid()}) running job {script_name}')
            start_time = datetime.now()
            results_item = script_name
            try:
                script_group, tasks, model_config, output_path, dataset_cache = args
                run_script(script_name, script_group, tasks, model_config, output_path, dataset_cache)
            except:
                the_type, the_value, the_traceback = sys.exc_info()
                print(f'ERROR -- {script_name} crashed continuing with other processes...')
                print('crash information')
                print(the_type)
                print(the_value)
                traceback.print_tb(the_traceback, file=sys.stdout)
                results_item = f'BAD_{results_item}'
                if not allow_crash:
                    # Can't pickle traceback so don't try to store this
                    results_item = (results_item, (the_type, the_value))
                temp = sys.stdout
                sys.stdout = sys.__stdout__
                print(f'ERROR -- {script_name} crashed (see log for info) continuing with other processes...')
                sys.stdout = temp
            finally:
                print('Experiment start time', start_time.strftime('%D %H:%M:%S'))
                print('Experiment end time', datetime.now().strftime('%D %H:%M:%S'))
                sys.stdout = sys.__stdout__
                # After finishing a job register it as finished
                self.results_queue.put(results_item)

def run_script(script_name, script_group, tasks, model_config, output_path, dataset_cache):
    torch.multiprocessing.set_sharing_strategy('file_system')
    config = ConfigClass({
        'model_save_dir': os.path.join(output_path, 'TrainedModels'),
        'tb_log_dir': os.path.join(output_path, 'tb_log_files'),
        'script_log_dir': os.path.join(output_path, 'logs'),
    })
    mkdirp(f"{config['script_log_dir']}/{script_group}")
    log_file = os.path.join(config['script_log_dir'], f'{script_group}/{script_name}.log')
    sys.stdout = sys.stderr = open(log_file, 'w', buffering=1)
    start_time = datetime.now()
    print('logging begun, current time', start_time.strftime('%D %H:%M:%S'))
    run_helper(script_name, script_group, tasks, config=config.config, model_config=model_config, dataset_cache=dataset_cache)

def run_experiments(jobs, output_path, dataset_cache_path, allow_crash=False):
    # Do this immediately to ensure that it isn't reset in other methods
    torch.multiprocessing.set_sharing_strategy('file_system')
    mp.set_start_method('spawn')
    # Add the output_path and dataset_cache_path to all jobs 
    for key in list(jobs.keys()):
        jobs[key] = tuple([*jobs[key], output_path, dataset_cache_path])
    # Define the config for use in the code
    config = ConfigClass({
        'model_save_dir': os.path.join(output_path, 'TrainedModels'),
        'tb_log_dir': os.path.join(output_path, 'tb_log_files'),
        'script_log_dir': os.path.join(output_path, 'logs'),
    })
    for d in config.config.values():
        mkdirp(d)

    # Before creating workers first make sure all datasets are precreated and cached 
    print('Creating and caching all datasets before multiprocessing to prevent race conditions')

    improv_path = dataset_cache_path
    ds = ImprovDatasetConstructor(dataset_save_location=improv_path)
    ds.close_h5()
    ds = PodcastDatasetConstructor(dataset_save_location=improv_path.replace('improv', 'podcast'))
    ds.close_h5()
    ds = IEMOCAPDatasetConstructor(dataset_save_location=improv_path.replace('improv', 'iemocap'))
    ds.close_h5()
    ds = MuSEDatasetConstructor(dataset_save_location=improv_path.replace('improv', 'muse'))
    ds.close_h5()
    print('Dataset verification complete')

    manager = WorkerManager()
    try:
        manager.create_workers()
        manager.launch_jobs(jobs, allow_crash=allow_crash)
        # manager.launch_jobs(initial_model_pretrain_configs, allow_crash=False)
        manager.wait_for_jobs(jobs.keys())
    finally:
        print('Closing processes..')
        manager.graceful_shutdown()
        for process in manager.processes:
            process.join()
        manager.job_queue.close()
        manager.results_queue.close()
        manager.job_queue.join_thread()
        manager.results_queue.join_thread()
