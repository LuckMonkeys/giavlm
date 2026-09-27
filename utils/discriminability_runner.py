"""Bounded argv-only scheduling for the private-reference study (physical GPUs 5/6)."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from core.artifacts import write_json
from utils.run_cmds import JobRunner, run_gpu


def validate_gpu_ids(value):
    ids = value.split(',') if isinstance(value, str) else [str(v) for v in value]
    if not ids or len(ids) > 2 or len(set(ids)) != len(ids) or not set(ids) <= {'5', '6'}:
        raise ValueError('Diagnostics allow only unique physical GPU IDs 5,6 (maximum two)')
    return ids


@contextmanager
def process_lock(name):
    root = Path('/tmp') / f'giavlm-discriminability-{os.getuid()}'
    root.mkdir(exist_ok=True, mode=0o700)
    with (root / f'{name}.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Diagnostic process lock is held: {name}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_inventory():
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total',
         '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in raw.strip().splitlines():
        index, uuid, free, total = [s.strip() for s in line.split(',')]
        result[index] = {'uuid': uuid, 'free_mib': int(free), 'total_mib': int(total)}
    return result


def worker_binding(device, tiny=False, inventory=None):
    """Validate visibility before model loading or any CUDA allocation."""
    if device == 'cpu':
        if not tiny:
            raise ValueError('CPU diagnostics are restricted to the tiny test model')
        return None
    if device != 'cuda:0':
        raise ValueError('A diagnostic worker must use its single visible device cuda:0')
    physical = os.environ.get('GIAVLM_DIAGNOSTIC_GPU', '')
    validate_gpu_ids([physical])
    devices = inventory if inventory is not None else gpu_inventory()
    if physical not in devices:
        raise ValueError('Assigned physical GPU is unavailable')
    uuid = devices[physical]['uuid']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('Worker must expose exactly its assigned GPU UUID')
    return {'physical_gpu': int(physical), 'uuid': uuid, 'pid': os.getpid()}


class DiagnosticRunner(JobRunner):
    def __init__(self, schedule, inventory, **kwargs):
        super().__init__(schedule, **kwargs)
        self.inventory = inventory
        original = self.popen

        def bounded_popen(argv, **options):
            env = options['env']
            physical = env['CUDA_VISIBLE_DEVICES']
            validate_gpu_ids([physical])
            env['GIAVLM_DIAGNOSTIC_GPU'] = physical
            env['CUDA_VISIBLE_DEVICES'] = self.inventory[physical]['uuid']
            env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            return original(argv, **options)

        self.popen = bounded_popen

    def start(self, job, gpu=None):
        running = super().start(job, gpu)
        if running is None:
            raise RuntimeError('Diagnostic worker failed to start')
        self.by_name[job['name']]['gpu_uuid'] = self.inventory[gpu]['uuid']
        self.by_name[job['name']]['pid'] = running['process'].pid
        self._commit()
        return running

    def reap(self):
        done = super().reap()
        if any(item['process'].returncode for item in done):
            raise RuntimeError('Diagnostic worker failed; remaining jobs must not continue')
        return done

    def terminate(self, grace_seconds=5.0):
        # Cancellation is already an error path; use base reap to finish every child.
        self.reap = lambda: JobRunner.reap(self)
        super().terminate(grace_seconds)


def dispatch(args, spec):
    ids = validate_gpu_ids(args.gpu_ids)
    resources = spec['resources']
    with process_lock('scheduler'):
        devices = gpu_inventory()
        if not set(ids) <= set(devices):
            raise ValueError('Requested diagnostic GPUs are not present')
        minimum = resources['fp32_min_free_mib' if args.precision == 'float32'
                            else 'min_free_mib']
        if all(devices[g]['total_mib'] < minimum for g in ids):
            raise RuntimeError('Requested experiment cannot fit its declared single-GPU budget')
        jobs = []
        conditions = [c for c in spec['conditions'] if not args.condition or
                      c['name'] == args.condition]
        if not conditions:
            raise ValueError('Unknown diagnostic condition')
        for condition in conditions:
            argv = [sys.executable, '-m', 'core.commands', '--threads', str(args.threads),
                    'diagnose-discriminability', '--spec', str(Path(args.spec).resolve()),
                    '--output', str(Path(args.output).resolve()), '--stage', args.stage,
                    '--cohort', args.cohort, '--precision', args.precision,
                    '--condition', condition['name'], '--device', 'cuda:0', '--worker']
            if args.resume:
                argv.append('--resume')
            jobs.append({'name': condition['name'], 'kind': 'command', 'argv': argv,
                         'min_free_mib': minimum})
        root = Path(args.output) / 'schedules' / args.cohort / args.precision / args.stage
        schedule = {'output_root': root, 'jobs': jobs,
                    'scheduler': {'min_free_mib': minimum, 'max_jobs_per_gpu': 1,
                                  'poll_seconds': resources['poll_seconds'],
                                  'occupancy': {'enabled': False}}}
        runner = DiagnosticRunner(schedule, devices)
        try:
            code = run_gpu(schedule, ids, runner)
            if code:
                raise RuntimeError(f'Diagnostic schedule failed: {code}')
        except BaseException as error:
            runner.terminate()
            write_json(root / 'failed.json', {'status': 'failed', 'error_type': type(error).__name__})
            raise
