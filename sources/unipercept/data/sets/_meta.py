"""Utility methods for generating a Metadata object."""

from __future__ import annotations

from typing import Sequence

from unicore.utils.frozendict import frozendict

from ..types import Metadata, SClass, StuffMode

__all__ = ["generate_metadata"]


def generate_metadata(
    sem_seq: Sequence[SClass],
    *,
    depth_max: float,
    label_divisor: int = 1000,
    ignore_depth: float = 0.0,
    ignore_label: int = 255,
    fps: float = 17.0,
    stuff_mode: StuffMode = StuffMode.ALL_CLASSES,
) -> Metadata:
    """Generate dataset metadata object."""

    sem_seq = sorted(
        # filter(
        #     lambda c: c["unified_id"] >= 0 and not c.is_void,
        #     sem_seq,
        # ),
        sem_seq,
        key=lambda c: (int(1e6) if c.get("is_thing") else 0) + c["unified_id"],
    )

    # Automatically resolves any many-to-one mappings for semantic IDs (only one semantic ID per metadata will remain)
    sem_map = {c["unified_id"]: c for c in sem_seq}

    # Offsets are the embedded channel index for either stuff or things
    stuff_offsets: dict[int, int] = {}
    for sem_id, sem_cls in sem_map.items():
        if not sem_cls.is_stuff:
            continue
        else:
            stuff_offsets[sem_id] = len(stuff_offsets)

    if stuff_mode == StuffMode.WITH_THING:
        # Cast to tuple in order to not alter the original dict while iterating
        for sem_id in tuple(stuff_offsets.keys()):
            stuff_offsets[sem_id] += 1

    thing_offsets: dict[int, int] = {}
    for sem_id, sem_cls in sem_map.items():
        if not sem_cls.is_thing:
            continue
        thing_offsets[sem_id] = len(thing_offsets)

        if stuff_mode == StuffMode.ALL_CLASSES:
            stuff_offsets[sem_id] = len(stuff_offsets)
        elif stuff_mode == StuffMode.WITH_THING:
            stuff_offsets[sem_id] = 0

    return Metadata(
        fps=fps,
        depth_max=depth_max,
        ignore_label=ignore_label,
        label_divisor=label_divisor,
        stuff_mode=stuff_mode,
        translations_dataset=frozendict({c["dataset_id"]: c["unified_id"] for c in sem_seq}),
        translations_semantic=frozendict({c["unified_id"]: c["dataset_id"] for c in sem_seq}),
        stuff_offsets=frozendict(stuff_offsets),
        thing_offsets=frozendict(thing_offsets),
        semantic_classes=frozendict(sem_map),
    )

    # num_thing = 0
    # num_stuff = 0
    # depth_fixed = {}
    # thing_classes = []
    # stuff_classes = []
    # thing_colors = []
    # stuff_colors = []
    # thing_translations = {}
    # thing_embeddings = {}
    # stuff_translations = {}
    # stuff_embeddings = {}
    # thing_train_id2contiguous_id = {}
    # stuff_train_id2contiguous_id = {}
    # cats_tracked = []

    # # Create definition of training IDs, which differ for stuff and things
    # for k in categories:
    #     dataset_id = k["id"]
    #     train_id = k["trainId"]
    #     if k["trainId"] == ignore_label:
    #         continue

    #     if bool(k.get("isthing")) == 1:
    #         id_duplicate = train_id in thing_embeddings
    #         thing_translations[dataset_id] = train_id

    #         if not id_duplicate:
    #             assert train_id not in stuff_embeddings, f"Train ID {train_id} is duplicated in stuff."

    #             thing_classes.append(k["name"])
    #             thing_colors.append(RGB(*k["color"]))
    #             thing_embeddings[train_id] = num_thing
    #             num_thing += 1

    #         if stuff_all_classes:
    #             stuff_translations[dataset_id] = train_id
    #             if not id_duplicate:
    #                 stuff_embeddings[train_id] = num_stuff
    #                 num_stuff += 1
    #     else:
    #         id_duplicate = train_id in stuff_embeddings
    #         if not id_duplicate:
    #             stuff_classes.append(k["name"])
    #             stuff_colors.append(RGB(*k["color"]))
    #         if stuff_with_things and not stuff_all_classes:
    #             stuff_translations[dataset_id] = train_id
    #             if not id_duplicate:
    #                 num_stuff += 1
    #                 stuff_embeddings[train_id] = num_stuff + 1
    #         else:
    #             stuff_translations[dataset_id] = train_id
    #             if not id_duplicate:
    #                 stuff_embeddings[train_id] = num_stuff
    #                 num_stuff += 1

    # # Sanity check
    # assert 0 in thing_embeddings.values()
    # assert 0 in stuff_embeddings.values()

    # # Create train ID to color mapping
    # colors: dict[int, RGB] = {}
    # colors |= {k: thing_colors[v] for k, v in thing_embeddings.items()}
    # colors |= {k: stuff_colors[v] for k, v in stuff_embeddings.items()}

    # # Create inverse translations
    # for key, value in thing_embeddings.items():
    #     thing_train_id2contiguous_id[value] = key
    # for key, value in stuff_embeddings.items():
    #     stuff_train_id2contiguous_id[value] = key

    # return Metadata(
    #     colors=frozendict(colors),
    #     stuff_all_classes=stuff_all_classes,
    #     stuff_with_things=stuff_with_things,
    #     ignore_label=ignore_label,
    #     fps=fps,
    #     num_thing=num_thing,
    #     num_stuff=num_stuff,
    #     label_divisor=label_divisor,
    #     depth_max=depth_max,
    #     depth_fixed=frozendict(depth_fixed),
    #     thing_classes=tuple(thing_classes),
    #     stuff_classes=tuple(stuff_classes),
    #     thing_colors=tuple(thing_colors),
    #     stuff_colors=tuple(stuff_colors),
    #     thing_translations=frozendict(thing_translations),
    #     thing_embeddings=frozendict(thing_embeddings),
    #     stuff_translations=frozendict(stuff_translations),
    #     stuff_embeddings=frozendict(stuff_embeddings),
    #     thing_train_id2contiguous_id=frozendict(thing_train_id2contiguous_id),
    #     stuff_train_id2contiguous_id=frozendict(stuff_train_id2contiguous_id),
    #     cats_tracked=frozenset(thing_train_id2contiguous_id.values()),
    # )
