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


class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.sorted_list = sorted(nums)
        if len(self.sorted_list) >= k:
            self.res = self.sorted_list[-k]
        else:
            self.res = 0
        self.k = k

    def add(self, val: int) -> int:
        idx = bisect.bisect_left(self.sorted_list, val)
        self.sorted_list.insert(idx, val)
        return self.sorted_list[-self.k]

    def print(self):
        print(self.sorted_list)
        print(self.res)


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

    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        left, right = 0, len(arr) - 1
        while left < right and arr[left] < arr[left + 1]:
            left += 1
        while right > 0 and arr[right - 1] < arr[right]:
            right -= 1
        if left == len(arr) - 1:
            return 0
        res = min(len(arr) - left - 1, right)

        for i in range(left + 1):
            if arr[i] <= arr[right]:
                res = min(res, right - i - 1)
            elif right < len(arr) - 1:
                right += 1
            else:
                break
        return res

    def lastStoneWeight(self, stones: List[int]) -> int:
        while len(stones) > 1:
            stones = sorted(stones)
            new_stones = stones[-1] - stones[-2]
            stones = stones[:-2]
            if new_stones:
                stones.append(new_stones)
        if len(stones):
            return stones[0]
        return 0

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        first = -1
        last = - 1
        flag = False
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if target == nums[mid]:
                flag = True
            if target <= nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        first = left

        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if target >= nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
        last = left - 1
        if not flag:
            return [-1, -1]
        return [first, last]


def main():
    nums = []
    target = 6
    test = Solution()
    res = test.searchRange(nums, target)
    print(res)


if __name__ == '__main__':
    main()
