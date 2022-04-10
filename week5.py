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
from collections import deque


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def print(self):
        s = []
        while self:
            s.append(str(self.val))
            self = self.next
        print(' -> '.join(s))


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


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

    def findKthPositive(self, arr: List[int], k: int) -> int:
        missing = 0
        prev = 0
        for i in range(0, len(arr)):
            curr = arr[i]
            temp = missing + (curr - prev - 1)
            if temp >= k:
                return prev + k - missing

            missing = temp
            prev = curr
        return arr[-1] + k - missing

    def arrangeCoins(self, n: int) -> int:
        left = 0
        right = n
        while left <= right:
            mid = left + (right - left) // 2
            if mid * (mid + 1) / 2 == n:
                return mid
            elif mid * (mid + 1) / 2 < n:
                left = mid + 1
            else:
                right = mid - 1
        return left - 1

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        num_count = collections.defaultdict(int)
        for num in nums:
            num_count[num] += 1
        sorted_dict = dict(sorted(num_count.items(), key=lambda x: x[1], reverse=True))
        return list(sorted_dict.keys())[:k]

    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) in [0, 1]:
            return len(s)
        res = 0
        left = 0
        right = 0
        seen = deque()
        while right < len(s):
            if s[right] in seen:
                res = max(res, right - left)
                while s[right] in seen:
                    seen.popleft()
                    left += 1
            seen.append(s[right])
            right += 1
        return max(res, len(seen))

    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False
        n = len(s1)
        s1_count = collections.Counter(s1)
        slide = s2[:n]
        s2_count = collections.Counter(slide)
        if s1_count == s2_count:
            return True
        for idx in range(1, len(s2) - n + 1):
            s2_count[s2[idx - 1]] -= 1
            if s2_count[s2[idx - 1]] == 0:
                del s2_count[s2[idx - 1]]
            s2_count[s2[idx + n - 1]] += 1
            if s1_count == s2_count:
                return True
        return False

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        marked = []
        res = 0

        def dfs(i, j):
            nonlocal marked, res
            temp = 0
            marked.append([i, j])
            if grid[i][j] == 0:
                return 0
            else:
                temp += 1
                if i - 1 >= 0 and [i - 1, j] not in marked:
                    temp += dfs(i - 1, j)
                if i + 1 < len(grid) and [i + 1, j] not in marked:
                    temp += dfs(i + 1, j)
                if j - 1 >= 0 and [i, j - 1] not in marked:
                    temp += dfs(i, j - 1)
                if j + 1 < len(grid[0]) and [i, j + 1] not in marked:
                    temp += dfs(i, j + 1)
            res = max(temp, res)
            return temp

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                dfs(row, col)
        return res

    def sortedArrayToBST(self, nums: List[int]) -> [TreeNode]:
        def recursion(left, right):
            if left > right:
                return None
            mid = left + (right - left) // 2
            root = TreeNode(nums[mid])
            root.left = recursion(left, mid - 1)
            root.right = recursion(mid + 1, right)
            return root

        return recursion(0, len(nums) - 1)

    def calPoints(self, ops: List[str]) -> int:
        stack = collections.deque()
        for item in ops:
            try:
                stack.append(int(item))
            except ValueError:
                if item == '+':
                    stack.append(stack[-1] + stack[-2])
                elif item == 'D':
                    stack.append(stack[-1] * 2)
                else:
                    stack.pop()
        return sum(stack)

    def countNegatives(self, grid: List[List[int]]) -> int:
        res = 0
        for m in range(len(grid)):
            left = 0
            right = len(grid[0]) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if grid[m][mid] >= 0:
                    left = mid + 1
                else:
                    right = mid - 1
            res += len(grid[m]) - left
        return res

    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        left = 0
        right = len(matrix) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if matrix[mid][0] > target:
                right = mid - 1
            else:
                left = mid + 1
        if left < 0:
            return False
        row_idx = left - 1

        left = 0
        right = len(matrix[0]) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if matrix[row_idx][mid] == target:
                return True
            elif matrix[row_idx][mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return False

    def mergeTrees(self, root1: [TreeNode], root2: [TreeNode]) -> [TreeNode]:
        if not root1:
            return root2
        if not root2:
            return root1

        def recursion(node1, node2):
            if node1 and node2:
                node1.val += node2.val
            elif not node1:
                return node2
            elif not node2:
                return node1
            elif not node1 and not node2:
                return None
            node1.left = recursion(node1.left, node2.left)
            node1.right = recursion(node1.right, node2.right)
            return node1

        root1 = recursion(root1, root2)
        return root1

    def connect(self, root):
        if not root:
            return None
        queue = collections.deque([root])
        while len(queue) > 0:
            temp = None
            for i in range(len(queue)):
                curr = queue.popleft()
                curr.next = temp
                temp = curr
                if curr.right:
                    queue.append(curr.right)
                    queue.append(curr.left)
        return root

    def reorderList(self, head: [ListNode]) -> None:
        if not head or not head.next:
            return head
        queue = collections.deque()
        temp = head
        while temp:
            queue.append(temp)
            temp = temp.next
        temp = head
        temp = queue.popleft()
        temp.next = queue.pop()
        temp = temp.next
        while len(queue) > 1:
            first = queue.popleft()
            last = queue.pop()
            last.next = None
            first.next = last
            temp.next = first
            temp = last
        if len(queue):
            temp.next = queue.pop()
            temp.next.next = None
        else:
            temp.next = None
        return head

    def reverseKGroup(self, head: [ListNode], k: int) -> [ListNode]:
        def reverse_list(node):
            if not head:
                return head

            def reverseHelper(node):
                if not node.next:
                    return node
                new_head = reverseHelper(node.next)
                node.next.next = node
                return new_head

            new_head = reverseHelper(head)
            head.next = None
            return new_head

        start = ListNode(next=head)
        end = start
        final = start
        while True:
            count = 0
            for i in range(k):
                end_ptr = end_ptr.next
                if end_ptr is None: break
            if end_ptr is None: break
            temp_next = end.next
            end.next = None
            reverse_list(start.next)
            start.next.next = temp_next
            temp = start
            start = start.next
            temp.next = end
            end = start
        return final.next

    def findTheWinner(self, n: int, k: int) -> int:
        friend_list = list(range(1, n + 1))
        idx = 0
        while len(friend_list) > 1:
            removed = friend_list.pop((idx + k - 1) % len(friend_list))
            idx = bisect.bisect_left(friend_list, removed)
        return friend_list[0]

    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        removed = []
        for idx, letter in enumerate(s):
            if letter == '(':
                stack.append([idx, letter])
            elif letter == ')':
                if len(stack) == 0:
                    removed.append(idx)
                else:
                    stack.pop(-1)
        if len(stack):
            removed += [idx for idx, letter in stack]
        for idx in sorted(removed, reverse=True):
            s = s[:idx] + s[idx + 1:]
        return s

    def arraySign(self, nums: List[int]) -> int:
        sign = 1
        for num in nums:
            if num == 0:
                return 0
            else:
                sign = -sign if num < 0 else sign
        return sign

    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        arr = sorted(arr)
        if len(arr) == 2:
            return True
        base = arr[1] - arr[0]
        for idx in range(1, len(arr)):
            if arr[idx] - arr[idx - 1] != base:
                return False
        return True

    def isHappy(self, n: int) -> bool:
        seen = set()
        while n > 1 and n not in seen:
            seen.add(n)
            new = sum([int(x) ** 2 for x in str(n)])
            n = new
        if n > 1:
            return False
        return True

    def preorder(self, root: 'Node') -> List[int]:
        res = []

        def recursion(node):
            nonlocal res
            if not node:
                return
            res.append(node.val)
            for child in node.children:
                recursion(child)

        recursion(root)
        return res

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        temp = []
        res = []
        n1, n2 = len(nums1), len(nums2)
        for idx in range(n2 - 1, -1, -1):
            while len(stack) and stack[-1] <= nums2[idx]:
                del stack[-1]
            if len(stack):
                temp.append(stack[-1])
            else:
                temp.append(-1)
            stack.append(nums2[idx])
        temp = temp[::-1]
        for num in nums1:
            res.append(temp[nums2.index(num)])
        return res

    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        if len(coordinates) == 2:
            return True
        start = coordinates[0]
        end = coordinates[-1]
        for coord in coordinates[1:-1]:
            if (coord[0] - start[0]) * (end[1] - start[1]) != (end[0] - start[0]) * (coord[1] - start[1]):
                return False
        return True

    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        prefix_sum = [0] + list(itertools.accumulate(arr))
        res = 0
        for i in range(1, len(prefix_sum)):
            count = 1
            while i + count <= len(prefix_sum):
                res += prefix_sum[i + count - 1] - prefix_sum[i - 1]
                count += 2
        return res

    def maximumWealth(self, accounts: List[List[int]]) -> int:
        res = 0
        for customer in accounts:
            res = max(res, sum(customer))
        return res

    def diagonalSum(self, mat: List[List[int]]) -> int:
        res = 0
        n = len(mat)
        for i in range(n // 2):
            res += mat[i][i] + mat[i][n - i - 1] + mat[n - i - 1][i] + mat[n - i - 1][n - i - 1]
        if n % 2 == 1:
            res += mat[n // 2][n // 2]
        return res

    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def array_to_tree(left, right):
            nonlocal preorder_index
            if left > right:
                return None
            root_value = preorder[preorder_index]
            root = TreeNode(root_value)

            preorder_index += 1
            root.left = array_to_tree(left, inorder_index_map[root_value] - 1)
            root.right = array_to_tree(inorder_index_map[root_value] + 1, right)

            return root

        preorder_index = 0

        inorder_index_map = {}
        for index, value in enumerate(inorder):
            inorder_index_map[value] = index

        return array_to_tree(0, len(preorder) - 1)

    def zigzagLevelOrder(self, root: [TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = collections.deque([root])
        res = []
        forward = True
        while len(queue) > 0:
            temp = []
            for i in range(len(queue)):
                curr = queue.popleft()
                temp.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            if forward:
                res.append(temp)
                forward = False
            else:
                res.append(reversed(temp))
                forward = True

        return res

    def rightSideView(self, root: [TreeNode]) -> List[int]:
        if not root:
            return []
        queue = collections.deque([root])
        res = []
        while len(queue):
            temp = []
            for i in range(len(queue)):
                curr = queue.popleft()
                temp.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            res.append(temp[-1])
        return res

    def pathSum(self, root: [TreeNode], targetSum: int) -> List[List[int]]:
        if not root:
            return []

        res = []
        path = []

        def recursion(node):
            nonlocal path, res
            if not node:
                return
            path.append(node.val)
            if not node.left and not node.right:
                if sum(path) == targetSum:
                    res.append(path)
            if node.left:
                recursion(node.left)
            if node.right:
                recursion(node.right)
            path = path[:-1]

        recursion(root)
        return res


def main():
    mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    test = Solution()
    res = test.diagonalSum(mat)
    print(res)


if __name__ == '__main__':
    main()
