import copy
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

def main():
    s = "deeedbbcccbdaa"
    k = 3
    test = Solution()
    res = test.removeDuplicates(s, k)
    print(res)


if __name__ == '__main__':
    main()
