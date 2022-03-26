import copy
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
    def oddEvenList(self, head: [ListNode]) -> [ListNode]:
        if not head:
            return None
        odd_node = head
        even_node = head.next
        even_node_head = head.next
        while even_node and even_node.next:
            odd_node.next = odd_node.next.next
            even_node.next = even_node.next.next
            odd_node = odd_node.next
            even_node = even_node.next

        odd_node.next = even_node_head
        return head

    def twoOutOfThree(self, nums1: list[int], nums2: list[int], nums3: list[int]) -> list[int]:
        set1 = set(nums1)
        set2 = set(nums2)
        set3 = set(nums3)
        return list((set1 & set2) | (set2 & set3) | (set3 & set1))

    def highestPeak(self, isWater: list[list[int]]) -> list[list[int]]:
        list_idx = []
        res = []
        for i in range(len(isWater)):
            temp = []
            for j in range(len(isWater[0])):
                if isWater[i][j] == 1:
                    temp.append(0)
                    list_idx.append([i, j])
                else:
                    temp.append(None)
            res.append(temp)
        curr_value = 1
        while len(list_idx) > 0:
            temp_list_idx = []
            for curr_idx in list_idx:
                row, col = curr_idx[0], curr_idx[1]
                if row > 0 and res[row - 1][col] is None:
                    res[row - 1][col] = curr_value
                    temp_list_idx.append([row - 1, col])
                if row < len(res) - 1 and res[row + 1][col] is None:
                    res[row + 1][col] = curr_value
                    temp_list_idx.append([row + 1, col])
                if col > 0 and res[row][col - 1] is None:
                    res[row][col - 1] = curr_value
                    temp_list_idx.append([row, col - 1])
                if col < len(res[0]) - 1 and res[row][col + 1] is None:
                    res[row][col + 1] = curr_value
                    temp_list_idx.append([row, col + 1])
            list_idx = temp_list_idx
            curr_value += 1
        return res

    def countServers(self, grid: list[list[int]]) -> int:
        # def check_connection(row, col, matrix):
        #     return (matrix[row, :] == 1).sum() > 1 or (matrix[:, col] == 1).sum() > 1
        #
        # count = 0
        # grid = np.array(grid)
        # for i in range(len(grid)):
        #     for j in range(len(grid[0])):
        #         if grid[i][j] == 1 and check_connection(i, j, grid):
        #             count += 1
        # return count

        row_count_dict = collections.defaultdict(int)
        col_count_dict = collections.defaultdict(int)
        m, n = len(grid), len(grid[0])
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    count += 1
                    row_count_dict[i] += 1
                    col_count_dict[j] += 1
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and row_count_dict[i] == 1 and col_count_dict[j] == 1:
                    count -= 1
        return count

    def largestSubmatrix(self, matrix: list[list[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        count_matrix = copy.deepcopy(matrix)
        for j in range(n):
            for i in range(1, m):
                if matrix[i][j] == 1:
                    count_matrix[i][j] = 1 + count_matrix[i - 1][j]
        res = 0
        for i in range(m):
            row = sorted(count_matrix[i], reverse=True)
            min_height = row[0]
            res = max(res, min_height * 1)
            for j in range(1, n):
                if row[j] != 0:
                    temp = min(min_height, row[j]) * (j + 1)
                    if temp > res:
                        res = temp
        return res

    def sumRootToLeaf(self, root: [TreeNode]) -> int:
        def dfs(node, path, res):
            path += str(node.val)
            if node.left is None and node.right is None:
                res.append(path)
                return
            if node.left:
                dfs(node.left, path, res)
            if node.right:
                dfs(node.right, path, res)

        res = []
        path = ''
        dfs(root, path, res)
        return sum([int(x, base=2) for x in res])

    def isSumEqual(self, firstWord: str, secondWord: str, targetWord: str) -> bool:
        def convert(word):
            alpha_dict = dict(zip(string.ascii_lowercase, list(range(26))))
            res = ''
            for letter in word:
                res += str(alpha_dict[letter])
            return int(res)

        return convert(firstWord) + convert(secondWord) == convert(targetWord)

    def removeDuplicates(self, s: str, k: int) -> str:
        # while True:
        #     count = 0
        #     if len(s) < k:
        #         break
        #     i = 0
        #     while i < len(s):
        #         if s[i: i + k] == s[i] * k:
        #             s = s[:i] + s[i + k:]
        #             count += 1
        #         i += 1
        #     if count == 0:
        #         break
        # return s

        # stack = []
        # for letter in s:
        #     stack.append(letter)
        #     if len(stack) >= k and stack[-k:] == [stack[-1]] * k:
        #         stack = stack[: -k]
        # return ''.join(stack)

        stack = []
        for letter in s:
            if not stack:
                stack.append([letter, 1])
                continue
            if stack[-1][0] == letter:
                stack[-1][1] += 1
            else:
                stack.append([letter, 1])
            if stack[-1][1] == k:
                del stack[-1]

        res = ''
        for letter in stack:
            res += letter[0] * letter[1]
        return res

    def search(self, nums: list[int], target: int) -> int:
        def binary_search(nums, target, start, end):
            if start >= end:
                return -1
            idx = (end - start) // 2 + start
            if nums[idx] == target:
                return idx
            elif nums[idx] < target:
                return binary_search(nums, target, idx + 1, end)
            else:
                return binary_search(nums, target, start, idx)

        res = binary_search(nums, target, 0, len(nums))
        return res

    def isValidBST(self, root: [TreeNode]) -> bool:
        def recursion(node, min_value=-1 * (2 ** 31), max_value=(2 ** 31)):
            if not node:
                return
            if node.val < min_value or node.val > max_value:
                return False
            left_check = recursion(node.left, min_value, node.val - 1)
            right_check = recursion(node.right, node.val + 1, max_value)
            return left_check and right_check

        return recursion(root)

    def findTarget(self, root: [TreeNode], k: int) -> bool:
        seen = dict()
        if not root:
            return False

        def recursion(node, k, seen):
            if not node:
                return False
            if k - node.val in seen:
                return True
            else:
                seen[node.val] = 1
            return recursion(node.left, k, seen) or recursion(node.right, k, seen)

        return recursion(root, k, seen)

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # if not root:
        #     return root
        # lowest = root
        #
        # def check_descendant(node, child_node):
        #     if node.val == child_node.val:
        #         return True
        #     else:
        #         return check_descendant(node.left, child_node) or check_descendant(node.right, child_node)
        #
        # def finder(node, node1, node2):
        #     nonlocal lowest
        #     if (check_descendant(node.left, node1) and check_descendant(node.right, node2)) or \
        #             (check_descendant(node.right, node1) and check_descendant(node.left, node2)):
        #         lowest = node
        #         return node
        #     if (node == node1 and check_descendant(node, node2)) or (node == node2 and check_descendant(node, node1)):
        #         lowest = node
        #         return node
        #     if check_descendant(node.left, node1) and check_descendant(node.left, node2):
        #         lowest = node.left
        #         finder(node.left, node1, node2)
        #     elif check_descendant(node.right, node1) and check_descendant(node.right, node2):
        #         lowest = node.right
        #         finder(node.right, node1, node2)
        #
        # return lowest

        def recursion(node, p, q):
            if p.val > node.val and q.val > node.val and node.right:
                node = node.right
                return recursion(node, p, q)
            if p.val < node.val and q.val < node.val and node.left:
                node = node.left
                return recursion(node, p, q)
            return node

        return recursion(root, p, q)

    def singleNumber(self, nums: List[int]) -> int:
        # seen = dict()
        # for num in nums:
        #     if num not in seen:
        #         seen[num] = 1
        #     else:
        #         seen[num] += 1
        # return list(seen.keys())[list(seen.values()).index(1)]

        res = nums[0]
        for num in nums[1:]:
            res = res ^ num
        return res

    def majorityElement(self, nums: List[int]) -> int:
        num_dict = dict()
        for num in nums:
            if num not in num_dict:
                num_dict[num] = 1
            else:
                num_dict[num] += 1
            if num_dict[num] > len(nums) / 2:
                return num


def main():
    nums = [3, 2, 3]
    test = Solution()
    res = test.majorityElement(nums)
    print(res)


if __name__ == '__main__':
    main()
