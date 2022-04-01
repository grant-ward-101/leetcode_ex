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

    def reverseString(self, s: List[str]) -> None:
        n = len(s)
        for i in range(n // 2):
            temp = s[i]
            s[i] = s[n - 1 - i]
            s[n - 1 - i] = temp

    def wordPattern(self, pattern: str, s: str) -> bool:
        word_list = s.split()
        if len(word_list) != len(pattern):
            return False
        temp = dict()
        seen = dict()
        for letter, word in zip(pattern, word_list):
            if word not in seen:
                seen[word] = letter
            elif seen[word] != letter:
                return False
            if letter not in temp:
                temp[letter] = word
            elif temp[letter] != word:
                return False
        return True

    def partitionLabels(self, s: str) -> list[int]:
        res = []
        while len(s) > 0:
            current_letter = s[0]
            idx = 0
            last_curr = s.rfind(current_letter)
            while idx < last_curr:
                temp_letter = s[idx]
                last_temp = s.rfind(temp_letter)
                if last_temp > last_curr:
                    last_curr = last_temp
                idx += 1
            res.append(idx + 1)
            s = s[idx + 1:]
        return res


def main():
    pattern = "abba"
    s = "dog cat cat dog"
    test = Solution()
    res = test.wordPattern(pattern, s)
    print(res)


if __name__ == '__main__':
    main()