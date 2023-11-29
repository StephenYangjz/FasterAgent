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
    
class EnhancedLS(Policy):
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
        return -len(seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].prompt) * (now - seq_group.arrival_time)
    
class EnhancedLS2(Policy):
    '''
    Polinomial of prompt length, using truncated gaussian for modeling the tail distribution of prompt length, E(c(x)-c| x>c)
    '''
    def __init__(self):
        self.a, self.b, self.c, self.d, self.e = -30322.848259864368, 2338.63345995253, -1.3777005397586924, 0.00039162829374907055, -4.257215529437982e-08

    def calc(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3 + self.e * x ** 4

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return -self.calc(len(seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].prompt))

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

# class Huristic(Policy):
#     '''
#     Longest API calling time first
#     '''

#     def get_priority(
#         self,
#         now: float,
#         seq_group: SequenceGroup,
#     ) -> float:
#         api_info = seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].api_info
#         call_times = [api.call_info['time'] for api in api_info.function_info.values()]
#         if len(call_times) == 0:
#             return (0, now - seq_group.arrival_time)
#         max_call_times =  max(call_times)
#         wait_time = now - seq_group.arrival_time
#         prompt_len = -len(seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].prompt)

#         return (max(call_times), )
class ToolNum(Policy):
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
            return now - seq_group.arrival_time
        return sum(call_times)/5000 + now - seq_group.arrival_time

class ToolNum2(Policy):
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
            return now - seq_group.arrival_time
        return sum(call_times)/1000 + now - seq_group.arrival_time
    
class Estimation(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        api_info = seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].api_info
        call_times = [api.call_info['time'] for api in api_info.function_info.values()]
        prompt_length = len(seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].prompt)
        done_api_calls = (len(api_info.conversation_history) - 2) / 2
        all_api_calls = 0.0738 * len(api_info.function_info) + 1.69
        estimated_api_calls = all_api_calls - done_api_calls
        average_call_time = sum(call_times) / len(call_times) / 1000
        estimated_api_time = estimated_api_calls * average_call_time
        generation_length = estimated_api_calls * 71 + 60 # * (done_api_calls == 0)
        generation_time = generation_length * 0.024 + 0.00011849 * prompt_length - 0.04940516
        prefilling_time = (8e-5 * prompt_length + 8e-3) * estimated_api_calls + 4e-5 * 71 * estimated_api_calls * (estimated_api_calls + 1)
        # Todo: consider API length
        total_time = generation_time + prefilling_time + estimated_api_time
        # print(total_time, generation_time, prefilling_time, estimated_api_time)
        return -total_time

class Estimation2(Policy):
    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        api_info = seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].api_info
        call_times = [api.call_info['time'] for api in api_info.function_info.values()]
        prompt_length = len(seq_group.seqs_dict[next(iter(seq_group.seqs_dict))].prompt)
        done_api_calls = (len(api_info.conversation_history) - 2) / 2
        all_api_calls = 0.0738 * len(api_info.function_info) + 1.69
        estimated_api_calls = all_api_calls - done_api_calls
        average_call_time = sum(call_times) / len(call_times) / 1000
        estimated_api_time = estimated_api_calls * average_call_time
        generation_length = estimated_api_calls * 71 + 60 # * (done_api_calls == 0)
        generation_time = generation_length * 1.46440962e-02 + 2.45917548e-05 * prompt_length * generation_length -0.36915662 # r^2 = 0.93
        prefilling_time = (8e-5 * prompt_length + 8e-3) * estimated_api_calls + 4e-5 * 71 * estimated_api_calls * (estimated_api_calls + 1)
        # Todo: consider API length
        total_time = generation_time + prefilling_time + estimated_api_time
        # print(total_time, generation_time, prefilling_time, estimated_api_time)
        return -total_time




class PolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
        'ls': LS,
        'las': LAS,
        'els': EnhancedLS,
        'els2': EnhancedLS2,
        'toolnum': ToolNum,
        'toolnum2': ToolNum2,
        "est": Estimation,
        "est2": Estimation2
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
