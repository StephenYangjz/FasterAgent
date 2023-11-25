from typing import List

from vllm.sequence import SequenceGroup


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: List[SequenceGroup],
    ) -> List[SequenceGroup]:
        # return sorted(
        #     seq_groups,
        #     key=lambda seq_group: self.get_priority(now, seq_group),
        #     reverse=True,
        # )
        res = sorted(
            seq_groups,
            key=lambda seq_group: self.get_priority(now, seq_group),
            reverse=True,
        )
        return res


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time


class LS(Policy):
    '''
    Shortest prompt first
    '''

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        # get first element in the dict
        # print(seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].prompt, len(seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].prompt))
        return -len(seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].prompt)
    
class LAS(Policy):
    '''
    Longest API calling time first
    '''

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        api_info = seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].api_info
        call_times = [api.call_info['time'] for api in api_info.function_info.values()]
        if len(call_times) == 0:
            return (0, now - seq_group.arrival_time)
        return (max(call_times), now - seq_group.arrival_time)


class PolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
        'ls': LS,
        'las': LAS,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
