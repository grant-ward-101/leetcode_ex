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


class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums = sorted(nums)
        for i in range(len(nums) - 1, 1, -1):
            if nums[i] < nums[i - 1] + nums[i - 2]:
                return nums[i] + nums[i - 1] + nums[i - 2]
        return 0

    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        res = -1
        temp_max = float('inf')
        for idx, point in enumerate(points):
            if point[0] == x or point[1] == y:
                if abs(point[0] - x) + abs(point[1] - y) < temp_max:
                    temp_max = abs(point[0] - x) + abs(point[1] - y)
                    res = idx
        return res

    def threeSumMulti(self, arr: List[int], target: int) -> int:
        # digit_count = collections.Counter(arr)
        arr = sorted(arr)
        res = 0
        for i in range(len(arr) - 2):
            left = i + 1
            right = len(arr) - 1
            while left < right:
                temp = arr[i] + arr[left] + arr[right]
                if temp == target:
                    if arr[left] != arr[right]:
                        temp_right = right
                        while arr[temp_right] == arr[right] and temp_right > left:
                            temp_right -= 1
                        temp_left = left
                        while arr[temp_left] == arr[left] and temp_left < right:
                            temp_left += 1
                        res += (right - temp_right) * (temp_left - left)
                        right = temp_right
                        left = temp_left
                    else:
                        res += int((right - left + 1) * (right - left) / 2)
                        break
                elif temp > target:
                    right -= 1
                else:
                    left += 1

        return res

    def magicalString(self, n: int) -> int:
        s = '122'
        flag = '1'
        idx = 2
        while len(s) <= n:
            s += int(s[idx]) * flag
            if flag == '1':
                flag = '2'
            else:
                flag = '1'
            idx += 1
        return s[:n].count('1')

    def licenseKeyFormatting(self, s: str, k: int) -> str:
        s = (s.replace('-', '')).upper()
        temp = len(s) % k
        remain = ''
        remain = s[:temp]
        s = s[temp:]
        res = ''
        for i in range(0, len(s), k):
            res += '-' + s[i:i + k]
        if len(remain):
            return remain + res
        return res[1:]

    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        comb = []

        def backtracking(n, k, idx=0, start=1):
            nonlocal res, comb
            if idx == k:
                res.append(comb)
            else:
                for i in range(start, n + 1):
                    comb.append(i)
                    backtracking(n, k, idx + 1, i + 1)
                    comb = comb[:-1]
        
        backtracking(n, k)
        return res


def main():
    n = 4
    k = 2
    test = Solution()
    res = test.combine(n ,k)
    print(res)


if __name__ == '__main__':
    main()
