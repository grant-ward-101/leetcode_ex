import bisect
import copy
import functools
import itertools
import math
import operator
import random
from typing import List
import numpy as np
import collections
import string


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def minSumOfLengths(self, arr: List[int], target: int) -> int:
        def left_min(nums, k):
            min_res = [float('inf')] * len(nums)
            curr_min = float('inf')
            curr_sum = 0
            left_pointer = 0
            for i in range(len(nums)):
                curr_sum += nums[i]
                while curr_sum > k:
                    curr_sum -= nums[left_pointer]
                    left_pointer += 1
                if curr_sum == k:
                    curr_min = min(curr_min, i - left_pointer + 1)
                min_res[i] = curr_min
            return min_res

        min_res = float('inf')
        left_mins = left_min(arr, target)
        right_mins = left_min(arr[::-1], target)
        for idx in range(len(arr) - 1):
            min_res = min(min_res, left_mins[idx] + right_mins[len(arr) - idx - 2])
        if min_res == float('inf'):
            return -1
        return min_res


def main():
    arr = [24, 1, 21, 1, 4, 3, 27, 7, 5, 1, 12, 1, 1, 43, 2, 5, 4, 54, 34]
    # arr = [4, 3, 2, 6, 2, 3, 4]
    target = 54
    test = Solution()
    res = test.minSumOfLengths(arr, target)
    print(res)


if __name__ == '__main__':
    main()
