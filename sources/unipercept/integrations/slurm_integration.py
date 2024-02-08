"""
SLURM scheduler integration.

Current implementation only supports reading the job configuration, and not interacting with the scheduler itself.
"""
from __future__ import annotations

import multiprocessing
import os
import typing as T

from unipercept.config import get_env


class SLURMEnvironment:
    """
    Sentinel class that provides access to the SLURM environment variables.
    """

    __slots__ = ()
    __obj__: T.Self | None = None

    def __new__(cls):
        if cls.__obj__ is None:
            cls.__obj__ = super().__new__(cls)
        return cls.__obj__

    @property
    def is_slurm_job(self) -> bool:
        return "SLURM_JOB_ID" in os.environ

    @property
    def mpi_type(self) -> str:
        return get_env(str, "SLURM_MPI_TYPE", default="")

    @property
    def step_id(self) -> str:
        return get_env(str, "SLURM_STEP_ID", default="")

    @property
    def step_gpus(self) -> str:
        return get_env(str, "SLURM_STEP_GPUS", default="")

    @property
    def nodeid(self) -> str:
        return get_env(str, "SLURM_NODEID", default="")

    @property
    def pmixp_abort_agent_port(self) -> str:
        return get_env(str, "SLURM_PMIXP_ABORT_AGENT_PORT", default="")

    @property
    def task_pid(self) -> str:
        return get_env(str, "SLURM_TASK_PID", default="")

    @property
    def prio_process(self) -> str:
        return get_env(str, "SLURM_PRIO_PROCESS", default="")

    @property
    def cpu_bind_verbose(self) -> str:
        return get_env(str, "SLURM_CPU_BIND_VERBOSE", default="")

    @property
    def submit_dir(self) -> str:
        return get_env(str, "SLURM_SUBMIT_DIR", default="")

    @property
    def ecplug(self) -> str:
        return get_env(str, "SLURM_ECPLUG", default="")

    @property
    def stepid(self) -> str:
        return get_env(str, "SLURM_STEPID", default="")

    @property
    def srun_comm_host(self) -> str:
        return get_env(str, "SLURM_SRUN_COMM_HOST", default="")

    @property
    def distribution(self) -> str:
        return get_env(str, "SLURM_DISTRIBUTION", default="")

    @property
    def ersbac(self) -> str:
        return get_env(str, "SLURM_ERSBAC", default="")

    @property
    def ersrun(self) -> str:
        return get_env(str, "SLURM_ERSRUN", default="")

    @property
    def step_gres(self) -> str:
        return get_env(str, "SLURM_STEP_GRES", default="")

    @property
    def procid(self) -> str:
        return get_env(str, "SLURM_PROCID", default="")

    @property
    def job_gid(self) -> str:
        return get_env(str, "SLURM_JOB_GID", default="")

    @property
    def cpu_bind(self) -> str:
        return get_env(str, "SLURM_CPU_BIND", default="")

    @property
    def nodename(self) -> str:
        return get_env(str, "SLURMD_NODENAME", default="")

    @property
    def job_end_time(self) -> str:
        return get_env(str, "SLURM_JOB_END_TIME", default="")

    @property
    def tasks_per_node(self) -> str:
        return get_env(str, "SLURM_TASKS_PER_NODE", default="")

    @property
    def nnodes(self) -> str:
        return get_env(str, "SLURM_NNODES", default="")

    @property
    def launch_node_ipaddr(self) -> str:
        return get_env(str, "SLURM_LAUNCH_NODE_IPADDR", default="")

    @property
    def get_user_env(self) -> str:
        return get_env(str, "SLURM_GET_USER_ENV", default="")

    @property
    def step_tasks_per_node(self) -> str:
        return get_env(str, "SLURM_STEP_TASKS_PER_NODE", default="")

    @property
    def job_start_time(self) -> str:
        return get_env(str, "SLURM_JOB_START_TIME", default="")

    @property
    def gpus(self) -> str:
        return get_env(str, "SLURM_GPUS", default="")

    @property
    def pmix_mapping_serv(self) -> str:
        return get_env(str, "SLURM_PMIX_MAPPING_SERV", default="")

    @property
    def cpus_per_gpu(self) -> int:
        return get_env(int, "SLURM_CPUS_PER_GPU", default=multiprocessing.cpu_count())

    @property
    def job_nodelist(self) -> str:
        return get_env(str, "SLURM_JOB_NODELIST", default="")

    @property
    def cluster_name(self) -> str:
        return get_env(str, "SLURM_CLUSTER_NAME", default="")

    @property
    def nodelist(self) -> str:
        return get_env(str, "SLURM_NODELIST", default="")

    @property
    def gpus_on_node(self) -> str:
        return get_env(str, "SLURM_GPUS_ON_NODE", default="")

    @property
    def ntasks(self) -> int:
        return get_env(int, "SLURM_NTASKS", default=1)

    @property
    def umask(self) -> str:
        return get_env(str, "SLURM_UMASK", default="")

    @property
    def job_cpus_per_node(self) -> int:
        return get_env(
            int,
            "SLURM_JOB_CPUS_PER_NODE",
            default=multiprocessing.cpu_count() / self.job_num_nodes,
        )

    @property
    def topology_addr(self) -> str:
        return get_env(str, "SLURM_TOPOLOGY_ADDR", default="")

    @property
    def debug(self) -> str:
        return get_env(str, "SLURMD_DEBUG", default="")

    @property
    def working_cluster(self) -> str:
        return get_env(str, "SLURM_WORKING_CLUSTER", default="")

    @property
    def step_nodelist(self) -> str:
        return get_env(str, "SLURM_STEP_NODELIST", default="")

    @property
    def job_name(self) -> str:
        return get_env(str, "SLURM_JOB_NAME", default="")

    @property
    def srun_comm_port(self) -> str:
        return get_env(str, "SLURM_SRUN_COMM_PORT", default="")

    @property
    def job_gpus(self) -> int:
        return get_env(int, "SLURM_JOB_GPUS", default=0)

    @property
    def jobid(self) -> str:
        return get_env(str, "SLURM_JOBID", default="")

    @property
    def conf(self) -> str:
        return get_env(str, "SLURM_CONF", default="")

    @property
    def job_qos(self) -> str:
        return get_env(str, "SLURM_JOB_QOS", default="")

    @property
    def topology_addr_pattern(self) -> str:
        return get_env(str, "SLURM_TOPOLOGY_ADDR_PATTERN", default="")

    @property
    def step_resv_ports(self) -> str:
        return get_env(str, "SLURM_STEP_RESV_PORTS", default="")

    @property
    def cpus_on_node(self) -> int:
        return get_env(int, "SLURM_CPUS_ON_NODE", default=0)

    @property
    def job_num_nodes(self) -> int:
        return get_env(int, "SLURM_JOB_NUM_NODES", default=0)

    @property
    def erlast(self) -> str:
        return get_env(str, "SLURM_ERLAST", default="")

    @property
    def job_uid(self) -> str:
        return get_env(str, "SLURM_JOB_UID", default="")

    @property
    def job_partition(self) -> str:
        return get_env(str, "SLURM_JOB_PARTITION", default="")

    @property
    def script_context(self) -> str:
        return get_env(str, "SLURM_SCRIPT_CONTEXT", default="")

    @property
    def cpu_bind_list(self) -> str:
        return get_env(str, "SLURM_CPU_BIND_LIST", default="")

    @property
    def job_user(self) -> str:
        return get_env(str, "SLURM_JOB_USER", default="")

    @property
    def nprocs(self) -> int:
        return get_env(int, "SLURM_NPROCS", default=int)

    @property
    def submit_host(self) -> str:
        return get_env(str, "SLURM_SUBMIT_HOST", default="")

    @property
    def job_account(self) -> str:
        return get_env(str, "SLURM_JOB_ACCOUNT", default="")

    @property
    def export_env(self) -> str:
        return get_env(str, "SLURM_EXPORT_ENV", default="")

    @property
    def step_launcher_port(self) -> str:
        return get_env(str, "SLURM_STEP_LAUNCHER_PORT", default="")

    @property
    def gtids(self) -> str:
        return get_env(str, "SLURM_GTIDS", default="")

    @property
    def job_id(self) -> str:
        return get_env(str, "SLURM_JOB_ID", default="")

    @property
    def cpu_bind_type(self) -> str:
        return get_env(str, "SLURM_CPU_BIND_TYPE", default="")

    @property
    def step_num_tasks(self) -> int:
        return get_env(int, "SLURM_STEP_NUM_TASKS", default=0)

    @property
    def eclibr(self) -> str:
        return get_env(str, "SLURM_ECLIBR", default="")

    @property
    def step_num_nodes(self) -> int:
        return get_env(int, "SLURM_STEP_NUM_NODES", default=0)

    @property
    def localid(self) -> str:
        return get_env(str, "SLURM_LOCALID", default="")
