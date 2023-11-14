import deqflow2  # https://github.com/kurt-stolle/deq-flow
from unimodels import deqflow

import unipercept as up
from unipercept.utils.config import get_project_name, get_session_name
from unipercept.utils.config._lazy import bind as B
from unipercept.utils.config._lazy import call as L

from .data._dvps import DATASET_INFO, DATASET_NAME, data

__all__ = ["model", "data", "trainer"]

trainer = B(up.trainer.Trainer)(
    config=L(up.trainer.config.TrainConfig)(
        project_name=get_project_name(__file__),
        session_name=get_session_name(__file__),
        train_batch_size=4,
        train_epochs=10,
        infer_batch_size=4,
        eval_epochs=1,
        save_epochs=1,
    ),
    optimizer=L(up.trainer.OptimizerFactory)(
        opt="adamw",
    ),
    scheduler=L(up.trainer.SchedulerFactory)(
        scd="poly",
        warmup_epochs=1,
    ),
    callbacks=[up.trainer.callbacks.FlowCallback, up.trainer.callbacks.ProgressCallback],
)

model = deqflow.DEQFlowWrapper(
    deqflow=deqflow2.DEQFlow(
        # TODO merge
    )
)
