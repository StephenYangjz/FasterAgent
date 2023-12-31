import time
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any

from vllm.config import CacheConfig, SchedulerConfig, PreemptionMode
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)
from vllm.agents.utils import input_prompt
from vllm.transformers_utils.tokenizer import detokenize_incrementally
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup],
        run_mode: int, # 0 generation, 1 prompt, 2 cross
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.run_mode = run_mode
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def __str__(self) -> str:
        return f"SchedulerOutputs: {self.scheduled_seq_groups}, {self.run_mode}, {self.num_batched_tokens}, {self.blocks_to_swap_in}, {self.blocks_to_swap_out}, {self.blocks_to_copy}, {self.ignored_seq_groups}"


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        tokenizer: Any,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(
            policy_name=scheduler_config.policy)
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window)

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []
        # Sequence groups in the API_BLOCKED state.
        self.blocked: List[SequenceGroup] = []

        self.blocks_to_swap_out: Dict[int, int] = {}
        self.tokenizer = tokenizer

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            # We need to reverse the list as we are removing elements
            # from it as we iterate over it. If we don't do it,
            # indices will get messed up and we will skip over elements.
            for seq_group in reversed(state_queue):
                if seq_group.request_id in request_ids:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.get_seqs():
                        if seq.is_finished():
                            continue
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.free_seq(seq)
                    request_ids.remove(seq_group.request_id)
                    if not request_ids:
                        return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped or self.blocked

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped) + len(self.blocked)

    def get_str_unfinished_seq_groups(self) -> str:
        return f"{len(self.waiting)=} {len(self.running)=} {len(self.swapped)=} {len(self.blocked)=}"

    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        if self.blocks_to_swap_out:
            blocks_to_swap_out.update(self.blocks_to_swap_out)
            self.blocks_to_swap_out = {}

        # Fix the current time.
        now = time.monotonic()

        # Move from blocked to waiting/running/swapped according to the configurations of scheduler.
        if self.blocked and not blocks_to_swap_out:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            seq_lens: List[int] = []
            for seq_group in list(self.blocked):
                seqs = seq_group.get_seqs(status=SequenceStatus.API_BLOCKED)
                assert len(seqs) == 1
                ready_num = 0
                for seq in seqs:
                    if seq.api_info.task.done():
                        ready_num += 1
                if ready_num == len(seqs):
                    if self.scheduler_config.use_cross:
                        assert self.scheduler_config.preemption_mode == PreemptionMode.SWAP
                        assert seq_group.num_seqs() == 1, (
                            "Now we only support blocked sequence group which have only one sequence.")
                        # raise NotImplementedError
                        # Do not sort blocked queue for now.
                        response_tokens = 0
                        for seq in seq_group.get_seqs():
                            if not seq.api_info.has_get_response:
                                seq.api_info.conversation_history.append(
                                seq.api_info.task.result())
                                response_tokens = self.tokenizer.encode(
                                input_prompt([seq.api_info.task.result()]))
                                seq.set_response_token_ids(response_tokens)
                                for token_id in response_tokens:
                                    seq.append_token_id(token_id, {token_id: 0})
                                    self._decode_sequence(seq, seq_group.sampling_params)
                                seq.api_info.has_get_response = True
                            else:
                                response_tokens = seq.get_response_token_ids()
                        num_cross_tokens = len(
                            response_tokens) + seq_group.get_seqs()[0].get_len()

                        # Check length. (Move to FINISHED_IGNORED and free if reaching prompt lnegth limit.)
                        if num_cross_tokens > self.prompt_limit:
                            logger.warning(
                                f"Reponse tokens plus kv cache ({num_cross_tokens} tokens) is too long"
                                f" and exceeds limit of {self.prompt_limit}")
                            for seq in seq_group.get_seqs():
                                seq.status = SequenceStatus.FINISHED_IGNORED
                                self.free_seq(seq)
                            ignored_seq_groups.append(seq_group)
                            self.blocked.remove(seq_group)
                            continue
                        # If the sequence group either cannot be swapped in or cannot be allocated, continue.
                        if not self.block_manager.can_allocate(seq_group):
                            continue
                        # If the number of batched tokens exceeds the limit, continue.
                        new_seq_lens = seq_lens + [num_cross_tokens]
                        num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                        if (num_batched_tokens > self.scheduler_config.max_num_batched_tokens):
                            continue
                        # The total number of sequences in the RUNNING state should not
                        # exceed the maximum number of sequences.
                        num_new_seqs = seq_group.get_max_num_running_seqs()
                        if (num_curr_seqs + num_new_seqs >
                            self.scheduler_config.max_num_seqs):
                            continue
                        num_paddings = num_batched_tokens - sum(new_seq_lens)
                        if num_paddings > self.scheduler_config.max_paddings:
                            continue
                        # Update statistics.
                        seq_lens = new_seq_lens
                        num_curr_seqs += num_new_seqs
                        # Move from blocked to running & modify state to running.
                        self.blocked.remove(seq_group)
                        self.running.append(seq_group)
                        for seq in seq_group.get_seqs():
                            seq.status = SequenceStatus.SWAPPED
                        # Swap in and append blocks.
                        self._swap_in(seq_group, blocks_to_swap_in)
                        for seq in seq_group.get_seqs():
                            seq.status = SequenceStatus.RUNNING
                        self._append_slots(seq_group, blocks_to_copy)
                        # Produce SchedulerOutputs
                        scheduled.append(seq_group)
                    else:
                        self.unblock(seq_group)
            if scheduled or ignored_seq_groups:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    run_mode=2,
                    num_batched_tokens=len(
                        seq_lens) * max(seq_lens) if len(seq_lens) > 0 else 0,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs


        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            seq_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            # by ZincCat: now we do
            self.waiting = self.policy.sort_by_priority(now, self.waiting)
            while self.waiting:
                seq_group = self.waiting[0]

                assert seq_group.num_seqs() == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {self.prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    continue

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    break

                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens = seq_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                if (num_batched_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    break
                seq_lens = new_seq_lens

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(seq_group)

            if scheduled or ignored_seq_groups:
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    run_mode=1,
                    num_batched_tokens=len(
                        seq_lens) * max(seq_lens) if len(seq_lens) > 0 else 0,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted and not blocks_to_swap_out:
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)

            while self.swapped:
                seq_group = self.swapped[0]
                # If the sequence group cannot be swapped in, stop.
                if not self.block_manager.can_swap_in(seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                seq_group = self.swapped.pop(0)
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slot(seq_group, blocks_to_copy)
                num_curr_seqs += num_new_seqs
                self.running.append(seq_group)

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            run_mode=0,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.run_mode == 1,
                is_cross=scheduler_outputs.run_mode == 2,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slots(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        preemption_mode = self.get_preemption_mode(seq_group, preemption_mode)
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, "Invalid preemption mode."

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def unblock(self,
                seq_group: SequenceGroup,
                preemption_mode: Optional[PreemptionMode] = None) -> None:
        self.blocked.remove(seq_group)

        assert len(seq_group.get_seqs()) == 1
        response_tokens = self.tokenizer.encode(input_prompt(
            [seq_group.get_seqs()[0].api_info.task.result()]))
        if len(response_tokens) + seq_group.get_seqs()[0].get_len() > self.prompt_limit:
            preemption_mode = PreemptionMode.RECOMPUTE
            seqs = seq_group.get_seqs(status=SequenceStatus.API_BLOCKED)
            for seq in seqs:
                self.block_manager.free(seq)

        if not preemption_mode and self.scheduler_config.preemption_mode:
            preemption_mode = self.scheduler_config.preemption_mode
        preemption_mode = self.get_preemption_mode(seq_group, preemption_mode)
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._unblock_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._unblock_by_swap(seq_group)
        else:
            assert False, "Invalid preemption mode."

    def block(self,
              seq_group: SequenceGroup,
              preemption_mode: Optional[PreemptionMode] = None) -> None:
        self.blocked.append(seq_group)
        self.running.remove(seq_group)
        seqs = seq_group.get_seqs(status=SequenceStatus.API_BLOCKED)
        for seq in seqs:
            seq.status = SequenceStatus.RUNNING
        if not preemption_mode and self.scheduler_config.preemption_mode:
            preemption_mode = self.scheduler_config.preemption_mode
        preemption_mode = self.get_preemption_mode(seq_group, preemption_mode)
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._block_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._block_by_swap(seq_group, self.blocks_to_swap_out)
        else:
            assert False, "Invalid preemption mode."
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        for seq in seqs:
            seq.status = SequenceStatus.API_BLOCKED

    def _block_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            self.block_manager.free(seq)

    def _unblock_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.API_BLOCKED)
        assert len(seqs) == 1
        for seq in seqs:
            seq.api_info.conversation_history.append(
                seq.api_info.task.result())
            new_prompt = input_prompt([seq.api_info.task.result()])
            new_prompt_token_ids = self.tokenizer.encode(new_prompt)
            for token_id in new_prompt_token_ids:
                seq.append_token_id(token_id, {token_id: 0})
                self._decode_sequence(seq, seq_group.sampling_params)
            seq.status = SequenceStatus.WAITING
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _block_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)

    def _unblock_by_swap(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.API_BLOCKED):
            seq.api_info.conversation_history.append(
                seq.api_info.task.result())
            seq.api_info.response_tokens = self.tokenizer.encode(
                input_prompt([seq.api_info.task.result()]))
            seq.append_token_id(seq.api_info.response_tokens[0], {
                                seq.api_info.response_tokens[0]: 0})
            self._decode_sequence(seq, seq_group.sampling_params)
            seq.api_info.response_next = 1 if len(
                seq.api_info.response_tokens) > 1 else -1
            seq.status = SequenceStatus.SWAPPED
        self.swapped.append(seq_group)

    def get_preemption_mode(self, seq_group: SequenceGroup,
                            preemption_mode: Optional[PreemptionMode] = None) -> PreemptionMode:
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        return preemption_mode

    def _decode_sequence(self, seq: Sequence, prms: SamplingParams) -> None:
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset,
         read_offset) = detokenize_incrementally(
             self.tokenizer,
             all_input_ids=seq.get_token_ids(),
             prev_tokens=seq.tokens,
             prefix_offset=seq.prefix_offset,
             read_offset=seq.read_offset,
             skip_special_tokens=prms.skip_special_tokens,
             spaces_between_special_tokens=prms.spaces_between_special_tokens,
        )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text
