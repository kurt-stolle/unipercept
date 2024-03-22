#!/usr/bin/env python
"""
Cleanup WandB artifacts that have many versions, keeping only the most recent ones.
"""
from __future__ import annotations

import argparse
import pprint

import wandb
from unipercept.integrations.wandb_integration import artifact_historic_delete

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup WandB artifacts")
    parser.add_argument("--name", type=str, required=True, help="Type of artifact")
    parser.add_argument("--type", type=str, required=True, help="Name of artifact")
    parser.add_argument(
        "--keep", default=1, type=int, help="Number of versions to keep"
    )

    args = parser.parse_args()

    api = wandb.Api()
    artifact = api.artifact(args.name, args.type)

    artifact_historic_delete(artifact, args.keep)
