import pytest
import torch

import unipercept as up


def test_panoptic_evaluators():
    pass


def test_depth_evaluator_functional():
    pred_depth = torch.randn(1, 64, 64)
    true_depth = torch.randn(1, 64, 64)

    losses = up.evaluators._depth._depth_metrics_single(pred=pred_depth, true=true_depth)

    print(losses)


def test_depth_evaluator_class():
    evaluator = up.evaluators.DepthEvaluator()


def test_flow_evaluators():
    pass


def test_tracking_evaluators():
    pass
