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


def main():
    x = 3
    y = 4
    points = [[1, 2], [3, 1], [2, 4], [2, 3], [4, 4]]
    test = Solution()
    res = test.nearestValidPoint(x, y, points)
    print(res)


if __name__ == '__main__':
    main()
