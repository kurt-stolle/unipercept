import deqflow2  # https://github.com/kurt-stolle/deq-flow
from unimodels import deqflow

import unipercept as up
from unipercept.config import get_project_name, get_session_name
from unipercept.config._lazy import bind as B
from unipercept.config._lazy import call as L

from ._dataset import DATASET_INFO, DATASET_NAME, data

__all__ = ["model", "data", "engine"]

engine = B(up.engine.Engine)(
    config=L(up.engine.params.EngineParams)(
        project_name=get_project_name(__file__),
        session_name=get_session_name(__file__),
        train_batch_size=4,
        train_epochs=10,
        infer_batch_size=4,
        eval_epochs=1,
        save_epochs=1,
    ),
    optimizer=L(up.engine.OptimizerFactory)(
        opt="adamw",
    ),
    scheduler=L(up.engine.SchedulerFactory)(
        scd="poly",
        warmup_epochs=1,
    ),
    callbacks=[up.engine.callbacks.FlowCallback, up.engine.callbacks.ProgressCallback],
)

model = deqflow.DEQFlowWrapper(
    deqflow=deqflow2.DEQFlow(
        # TODO merge
    )
)
