import bisect
import copy
import functools
import itertools
import math
import operator
import random
import time
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


class larger_num(str):
    def __lt__(x, y):
        return x + y > y + x


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

    def largestNumber(self, nums: List[int]) -> str:
        largest_num = ''.join(sorted(map(str, nums), key=larger_num))
        return '0' if largest_num[0] == '0' else largest_num

    def minSubarray(self, nums: List[int], p: int) -> int:
        # total_sum = sum(nums)
        # if total_sum % p == 0:
        #     return 0
        #
        # prefix_sum = [0] + list(itertools.accumulate(nums))
        # postfix_sum = ([0] + list(itertools.accumulate(nums[::-1])))[::-1]
        # res = float('inf')
        # for i in range(len(prefix_sum)):
        #     for j in range(i, len(postfix_sum)):
        #         if prefix_sum[i] + postfix_sum[j] > 0:
        #             if (prefix_sum[i] + postfix_sum[j]) % p == 0:
        #                 res = min(res, j - i)
        #
        # if res == float('inf'):
        #     return -1
        # return res

        total_sum = sum(nums) % p
        if total_sum == 0:
            return 0
        n = len(nums)
        res = n
        remain = {0: -1}
        current_sum = 0
        for idx, value in enumerate(nums):
            current_sum = (current_sum + value) % p
            temp = (current_sum - total_sum) % p
            if temp in remain:
                res = min(res, idx - remain[temp])
            remain[current_sum] = idx
        if res == n:
            return -1
        else:
            return res

    def maxAbsValExpr(self, arr1: List[int], arr2: List[int]) -> int:
        a, b, c, d = [], [], [], []
        for i in range(len(arr1)):
            a.append(i + arr1[i] + arr2[i])
            b.append(i + arr1[i] - arr2[i])
            c.append(i - arr1[i] + arr2[i])
            d.append(i - arr1[i] - arr2[i])
        return max(max(a) - min(a), max(b) - min(b), max(c) - min(c), max(d) - min(d))


def main():
    arr1 = [1, 2, 3, 4]
    arr2 = [-1, 4, 5, 6]
    test = Solution()
    res = test.maxAbsValExpr(arr1, arr2)
    print(res)


if __name__ == '__main__':
    main()
