"""
Dumps things to tensorboard and console
"""

import datetime
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from pytz import timezone
from torch.utils.tensorboard import SummaryWriter

from c2ot.utils.time_estimator import PartialTimeEstimator, TimeEstimator
from c2ot.utils.timezone import my_timezone


class TensorboardLogger:

    def __init__(self,
                 exp_id: str,
                 run_dir: Union[Path, str],
                 py_logger: logging.Logger,
                 *,
                 is_rank0: bool = False):
        self.exp_id = exp_id
        self.run_dir = Path(run_dir)
        self.py_log = py_logger
        if is_rank0:
            self.tb_log = SummaryWriter(run_dir)
        else:
            self.tb_log = None

        # Get current git info for logging
        try:
            import git
            repo = git.Repo(".")
            git_info = str(repo.active_branch) + ' ' + str(repo.head.commit.hexsha)
        except (ImportError, RuntimeError, TypeError):
            print('Failed to fetch git info. Defaulting to None')
            git_info = 'None'

        self.log_string('git', git_info)

        # log the SLURM job id if available
        job_id = os.environ.get('SLURM_JOB_ID', None)
        if job_id is not None:
            self.log_string('slurm_job_id', job_id)

        # used when logging metrics
        self.batch_timer: TimeEstimator = None
        self.data_timer: PartialTimeEstimator = None

    def log_scalar(self, tag: str, x: float, it: int):
        if self.tb_log is None:
            return
        self.tb_log.add_scalar(tag, x, it)

    def log_metrics(self,
                    prefix: str,
                    metrics: dict[str, float],
                    it: int,
                    ignore_timer: bool = False):
        msg = f'{self.exp_id}-{prefix} - it {it:6d}: '
        metrics_msg = ''
        for k, v in sorted(metrics.items()):
            self.log_scalar(f'{prefix}/{k}', v, it)
            metrics_msg += f'{k: >10}:{v:.7f},\t'

        if self.batch_timer is not None and not ignore_timer:
            self.batch_timer.update()
            avg_time = self.batch_timer.get_and_reset_avg_time()
            data_time = self.data_timer.get_and_reset_avg_time()

            # add time to tensorboard
            self.log_scalar(f'{prefix}/avg_time', avg_time, it)
            self.log_scalar(f'{prefix}/data_time', data_time, it)

            est = self.batch_timer.get_est_remaining(it)
            est = datetime.timedelta(seconds=est)
            if est.days > 0:
                remaining_str = f'{est.days}d {est.seconds // 3600}h'
            else:
                remaining_str = f'{est.seconds // 3600}h {(est.seconds%3600) // 60}m'
            eta = datetime.datetime.now(timezone(my_timezone)) + est
            eta_str = eta.strftime('%Y-%m-%d %H:%M:%S %Z%z')
            time_msg = f'avg_time:{avg_time:.3f},data:{data_time:.3f},remaining:{remaining_str},eta:{eta_str},\t'
            msg = f'{msg} {time_msg}'

        msg = f'{msg} {metrics_msg}'
        self.py_log.info(msg)

    def log_image(self, prefix: str, tag: str, image: np.ndarray, it: int):
        image_dir = self.run_dir / f'{prefix}_images'
        image_dir.mkdir(exist_ok=True, parents=True)

        image = Image.fromarray(image)
        image.save(image_dir / f'{it:09d}_{tag}.png')

    def log_string(self, tag: str, x: str):
        self.py_log.info(f'{tag} - {x}')
        if self.tb_log is None:
            return
        self.tb_log.add_text(tag, x)

    def debug(self, x):
        self.py_log.debug(x)

    def info(self, x):
        self.py_log.info(x)

    def warning(self, x):
        self.py_log.warning(x)

    def error(self, x):
        self.py_log.error(x)

    def critical(self, x):
        self.py_log.critical(x)
