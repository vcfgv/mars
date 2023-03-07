# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Iterable
import functools
import logging
import numpy as np
import pandas as pd

from .....core.operand import MapReduceOperand, OperandStage
from .....utils import lazy_import
from ....subtask import Subtask, SubtaskGraph

ray = lazy_import("ray")
logger = logging.getLogger(__name__)


class ShuffleManager:
    """Manage shuffle execution for ray by resolve dependencies between mappers outputs and reducers inputs based on
    mapper and reducer index.
    """

    def __init__(self, subtask_graph: SubtaskGraph, config):
        self._subtask_graph = subtask_graph
        self._proxy_subtasks = subtask_graph.get_shuffle_proxy_subtasks()
        self._num_shuffles = subtask_graph.num_shuffles()
        self._mapper_output_refs = []
        self._merger_output_refs = []

        # TODO: Scheduler of rounds
        self._merging_factor = config.get("merging_factor") or 2
        self._roundwise_num_map_tasks = []
        self._current_round = 0
        self._current_shuffle = 0
        # recorded each round
        self._map_subtask_submit_history = []
        # cleared after one round
        self._submitted_map_subtask_indices = []
        self._map_subtasks = []
        self._max_concurrent_rounds = max(config.get("max_concurrent_rounds") or 2, 1)
        self._wait_timeout = max(config.get("wait_timeout") or 1, 1)
        # TODO: destroy()
        self._ray_merge_executor = self._get_ray_merge_executor()

        self._mapper_indices = {}
        self._reducer_indices = {}
        for shuffle_index, proxy_subtask in enumerate(self._proxy_subtasks):
            # Note that the reducers can also be mappers such as `DuplicateOperand`.
            mapper_subtasks = subtask_graph.predecessors(proxy_subtask)
            reducer_subtasks = subtask_graph.successors(proxy_subtask)

            n_mappers = len(mapper_subtasks)
            n_reducers = proxy_subtask.chunk_graph.results[0].op.n_reducers
            n_mergers = n_reducers
            n_rounds = min(self._merging_factor, n_mappers)

            even_parts = [n_mappers // n_rounds for _ in range(n_rounds)]
            for i in range(n_mappers % n_rounds):
                even_parts[i] += 1
            self._roundwise_num_map_tasks.append(even_parts)

            mapper_output_arr = np.empty((n_mappers, n_mergers), dtype=object)
            self._mapper_output_refs.append(mapper_output_arr)
            self._mapper_indices.update(
                {
                    subtask: (shuffle_index, mapper_index)
                    for mapper_index, subtask in enumerate(mapper_subtasks)
                }
            )
            merger_output_arr = np.empty((n_rounds, n_reducers), dtype=object)
            self._merger_output_refs.append(merger_output_arr)
            # reducers subtask should be sorted by reducer_index and MapReduceOperand.map should insert shuffle block
            # in reducers order, otherwise shuffle blocks will be sent to wrong reducers.
            sorted_filled_reducer_subtasks = self._get_sorted_filled_reducers(
                reducer_subtasks, n_reducers
            )
            self._reducer_indices.update(
                {
                    subtask: (shuffle_index, reducer_ordinal)
                    for reducer_ordinal, subtask in enumerate(
                        sorted_filled_reducer_subtasks
                    )
                }
            )

    @staticmethod
    def _get_sorted_filled_reducers(
        reducer_subtasks: Iterable[Subtask], n_reducers: int
    ):
        # For operands such as `PSRSAlign`, sometimes `reducer_subtasks` might be less than `n_reducers`.
        # fill missing reducers with `None`.
        filled_reducers = [None] * n_reducers
        for subtask in reducer_subtasks:
            reducer_ordinal = _get_reducer_operand(subtask.chunk_graph).reducer_ordinal
            filled_reducers[reducer_ordinal] = subtask
        return filled_reducers

    @staticmethod
    @functools.lru_cache(maxsize=None)  # Specify maxsize=None to make it faster
    def _get_ray_merge_executor():
        # Export remote function once.
        return ray.remote(execute_merge_task)

    def add_map_subtask(self, subtask, ray_options, ray_args):
        self._map_subtasks.append((subtask, ray_options, ray_args))
        num_map_tasks_current_round = self._roundwise_num_map_tasks[
            self._current_shuffle
        ][self._current_round]
        if len(self._map_subtasks) == num_map_tasks_current_round:
            self._submit_curr_map_subtasks()
            self._submit_merger_tasks()
            self._current_round += 1
            if self._current_round == len(
                self._roundwise_num_map_tasks[self._current_shuffle]
            ):
                logger.info(
                    "Rounds for current shuffle %s finished.", self._current_shuffle
                )
                self._current_round = 0
                self._current_shuffle += 1

    def _submit_curr_map_subtasks(self):
        for subtask, options, args in self._map_subtasks:
            n_mergers = self.get_n_reducers(subtask)
            output_object_refs = options.remote(*args)
            if not isinstance(output_object_refs, list):
                output_object_refs = [output_object_refs]
            self.add_mapper_output_refs(subtask, output_object_refs[-n_mergers:])
            _, mapper_index = self._mapper_indices[subtask]
            self._submitted_map_subtask_indices.append(mapper_index)
        logger.info(
            "[R%s S%s] Submitted %s map subtasks: %s",
            self._current_round,
            self._current_shuffle,
            len(self._submitted_map_subtask_indices),
            self._submitted_map_subtask_indices,
        )
        self._map_subtasks.clear()

    def _submit_merger_tasks(self):
        if self._current_round > self._max_concurrent_rounds - 1:
            prev_round = self._current_round - self._max_concurrent_rounds
            remaining = self._merger_output_refs[self._current_shuffle][
                prev_round
            ].tolist()
            while len(remaining) > 0:
                _, remaining = ray.wait(
                    remaining, fetch_local=False, timeout=self._wait_timeout
                )
            logger.info(
                "[R%s S%s] Tasks of round %s finished.",
                self._current_round,
                self._current_shuffle,
                prev_round,
            )
            for mapper_index in self._map_subtask_submit_history[prev_round]:
                self._mapper_output_refs[self._current_shuffle][mapper_index].fill(None)

        for merger_index, input_refs in enumerate(self._get_merger_input_refs()):
            output_object_ref = self._ray_merge_executor.options(
                num_returns=1,
            ).remote(merger_index, *input_refs)
            self._merger_output_refs[self._current_shuffle][self._current_round][
                merger_index
            ] = output_object_ref
        logger.info(
            "[R%s S%s] Submitted merger tasks.",
            self._current_round,
            self._current_shuffle,
        )
        self._map_subtask_submit_history.append(self._submitted_map_subtask_indices[:])
        self._submitted_map_subtask_indices.clear()

    def has_shuffle(self):
        """
        Whether current subtask graph has shuffles to execute.
        """
        return self._num_shuffles > 0

    def add_mapper_output_refs(
        self, subtask: Subtask, output_object_refs: List["ray.ObjectRef"]
    ):
        """
        Record mapper output ObjectRefs which will be used by reducers later.

        Parameters
        ----------
        subtask
        output_object_refs : List["ray.ObjectRef"]
            Mapper output ObjectRefs.
        """
        shuffle_index, mapper_index = self._mapper_indices[subtask]
        self._mapper_output_refs[shuffle_index][mapper_index] = np.array(
            output_object_refs
        )

    def _get_merger_input_refs(self) -> List["ray.ObjectRef"]:
        shuffle_index = self._current_shuffle
        mapper_indices = self._submitted_map_subtask_indices
        num_mergers = self._mapper_output_refs[shuffle_index].shape[1]
        merger_input_refs = np.empty((num_mergers, len(mapper_indices)), dtype=object)
        for merger_index in range(num_mergers):
            for idx, mapper_index in enumerate(mapper_indices):
                merger_input_refs[merger_index][idx] = self._mapper_output_refs[
                    shuffle_index
                ][mapper_index][merger_index]
        return merger_input_refs.tolist()

    def get_reducer_input_refs(self, subtask: Subtask) -> List["ray.ObjectRef"]:
        """
        Get the reducer inputs ObjectRefs output by mappers.

        Parameters
        ----------
        subtask : Subtask
            A reducer subtask.
        Returns
        -------
        input_refs : List["ray.ObjectRef"]
            The reducer inputs ObjectRefs output by mappers.
        """
        from .executor import _get_fetch_chunks

        shuffle_index, reducer_ordinal = self._reducer_indices[subtask]
        reducer_input_refs = self._merger_output_refs[shuffle_index][:, reducer_ordinal]

        _, shuffle_fetch_chunk = _get_fetch_chunks(subtask.chunk_graph)
        shuffle_fetch_chunk.op.n_mappers = len(reducer_input_refs)
        return reducer_input_refs

    def get_n_reducers(self, subtask: Subtask):
        """
        Get the number of shuffle blocks that a mapper operand outputs,
        which is also the number of the reducers when tiling shuffle operands.
        Note that this might be greater than actual number of the reducers in the subtask graph,
        because some reducers may not be added to chunk graph.

        Parameters
        ----------
        subtask : Subtask
            A mapper or reducer subtask.
        Returns
        -------
        n_reducers : int
            The number of shuffle blocks that a mapper operand outputs.
        """
        index = self._mapper_indices.get(subtask) or self._reducer_indices.get(subtask)
        if index is None:
            raise ValueError(f"The {subtask} should be a mapper or a reducer.")
        else:
            shuffle_index, _ = index
            return self._mapper_output_refs[shuffle_index].shape[1]

    def is_mapper(self, subtask: Subtask):
        """
        Check whether a subtask is a mapper subtask. Note the even this a mapper subtask, it can be a reducer subtask
        at the same time such as `DuplicateOperand`, see
        https://user-images.githubusercontent.com/12445254/174305282-f7c682a9-0346-47fe-a34c-1e384e6a1775.svg
        """
        return subtask in self._mapper_indices

    def info(self):
        """
        A list of (mapper count, reducer count).
        """
        return [shuffle_mapper.shape for shuffle_mapper in self._mapper_output_refs]

    def remove_object_refs(self, subtask: Subtask):
        """
        Set the object refs to None by subtask.
        """
        index = self._mapper_indices.get(subtask)
        if index is not None:
            shuffle_index, mapper_index = index
            self._mapper_output_refs[shuffle_index][mapper_index].fill(None)
            return
        index = self._reducer_indices.get(subtask)
        if index is not None:
            shuffle_index, reducer_ordinal = index
            self._mapper_output_refs[shuffle_index][:, reducer_ordinal].fill(None)
            self._merger_output_refs[shuffle_index][:, reducer_ordinal].fill(None)
            return
        raise ValueError(f"The {subtask} should be a mapper or a reducer.")


def _get_reducer_operand(subtask_chunk_graph):
    return next(
        c.op
        for c in subtask_chunk_graph
        if isinstance(c.op, MapReduceOperand) and c.op.stage == OperandStage.reduce
    )


def execute_merge_task(merge_index, *inputs):
    # Vanilla
    sample = inputs[0]
    if isinstance(sample, (pd.DataFrame, pd.Series)):
        output = pd.concat(inputs)
    elif isinstance(sample, np.ndarray):
        output = np.concatenate(inputs)
    else:
        # Tuple[Tuple(chunk index), pandas.DataFrame]
        # Tuple[mapper_id, Tuple(chunk index), pd.DataFrame]
        output = list(inputs)

    return output
