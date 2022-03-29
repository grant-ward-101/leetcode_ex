import copy
import math
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


class MyHashMap:

    def __init__(self):
        self.key = []
        self.value = []

    def put(self, key: int, value: int) -> None:
        if key not in self.key:
            self.key.append(key)
            self.value.append(value)
        else:
            self.value[self.key.index(key)] = value

    def get(self, key: int) -> int:
        if key not in self.key:
            return -1
        else:
            return self.value[self.key.index(key)]

    def remove(self, key: int) -> None:
        if key in self.key:
            idx = self.key.index(key)
            del self.key[idx]
            del self.value[idx]


class RandomizedSet:

    def __init__(self):
        self.randomized_set = collections.defaultdict(int)

    def insert(self, val: int) -> bool:
        if val not in self.randomized_set:
            self.randomized_set[val] = 1
            return True
        return False

    def remove(self, val: int) -> bool:
        if val in self.randomized_set:
            del self.randomized_set[val]
            return True
        return False

    def getRandom(self) -> int:
        choice = random.choice(list(self.randomized_set.keys()))
        return choice


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

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # res = []
        # for i in range(len(nums)):
        #     num1 = nums[i]
        #     remain = nums[:i] + nums[i + 1:]
        #     seen = dict()
        #     for j in range(len(remain)):
        #         num2 = remain[j]
        #         num3 = 0 - num1 - num2
        #         if num3 in seen:
        #             temp = sorted([num1, num2, num3])
        #             if temp not in res:
        #                 res.append(temp)
        #         else:
        #             seen[num2] = 1
        # return res

        res = []
        nums = sorted(nums)
        for i in range(len(nums) - 2):
            sum_required = -nums[i]
            j = i + 1
            k = len(nums) - 1
            while j < k:
                if nums[j] + nums[k] == sum_required:
                    if [nums[i], nums[j], nums[k]] not in res:
                        res.append([nums[i], nums[j], nums[k]])

                if nums[j] + nums[k] < sum_required:
                    j += 1
                else:
                    k -= 1
        return res

    def frequencySort(self, nums: List[int]) -> List[int]:
        nums = sorted(nums)
        counter = collections.Counter(nums)
        res = []
        temp = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for i, j in reversed(temp):
            res += [i] * j
        return res

    def sortEvenOdd(self, nums: List[int]) -> List[int]:
        temp = np.zeros(len(nums))
        nums = np.array(nums)
        even_idx = np.arange(0, len(nums), 2)
        odd_idx = np.arange(1, len(nums), 2)
        even = np.sort(nums[even_idx])
        odd = np.sort(nums[odd_idx])
        temp[even_idx] = even
        temp[odd_idx] = odd
        res = temp.astype(int).tolist()
        return res

    def sortArray(self, nums: List[int]) -> List[int]:
        def quick_sort(nums, start, end):
            if start >= end:
                return
            pivot_idx = (start + end) // 2
            pivot = nums[pivot_idx]
            left, right = start, end
            while left <= right:
                while left <= right and nums[left] < pivot:
                    left += 1
                while left <= right and nums[right] > pivot:
                    right -= 1
                if left <= right:
                    nums[left], nums[right] = nums[right], nums[left]
                    left += 1
                    right -= 1
            quick_sort(nums, start, right)
            quick_sort(nums, left, end)

        quick_sort(nums, 0, len(nums) - 1)
        return nums

    def isSubsequence(self, s: str, t: str) -> bool:
        i = 0
        j = 0
        while i < len(s):
            if j >= len(t):
                return False
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                j += 1

        return True

    def mostVisited(self, n: int, rounds: List[int]) -> List[int]:
        count_dict = dict(zip(list(range(1, n + 1)), [0] * n))
        last_stop = None
        for i in range(1, len(rounds)):
            if last_stop is not None:
                count_dict[last_stop] -= 1
            track_start = rounds[i - 1]
            track_end = rounds[i]
            last_stop = track_end
            track_list = []
            if track_end <= track_start:
                track_list = track_list + list(range(track_start, n + 1)) + list(range(1, track_end + 1))
            else:
                track_list = track_list + list(range(track_start, track_end + 1))

            for sector in track_list:
                count_dict[sector] += 1
        res = [key for key, value in count_dict.items() if value == max(count_dict.values())]
        return sorted(res)

    def countQuadruplets(self, nums: List[int]) -> int:
        res = 0
        for i in range(len(nums) - 3):
            for j in range(i + 1, len(nums) - 2):
                for k in range(j + 1, len(nums) - 1):
                    sum = nums[i] + nums[j] + nums[k]
                    res += nums[k + 1:].count(sum)
        return res

    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        def calc_dist(point1, point2):
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        res = 0
        for i in range(len(points)):
            dist_dict = collections.defaultdict(int)
            for j in range(len(points)):
                if i != j:
                    temp_dist = calc_dist(points[i], points[j])
                    dist_dict[temp_dist] += 1
            for item in dist_dict:
                res += dist_dict[item] * (dist_dict[item] - 1)
        return res

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        def check_valid(row, col, grid, valid_list):
            check_pacific = False
            check_atlantic = False
            marked = []
            index_list = [[row, col]]
            while len(index_list) > 0:
                curr_loc = index_list[0]
                if curr_loc in valid_list:
                    return True
                if curr_loc[0] == 0 or curr_loc[1] == 0:
                    check_pacific = True
                if curr_loc[0] == len(grid) - 1 or curr_loc[1] == len(grid[0]) - 1:
                    check_atlantic = True
                up = [curr_loc[0] - 1, curr_loc[1]]
                left = [curr_loc[0], curr_loc[1] - 1]
                if up not in index_list and up not in marked and up[0] >= 0 and \
                        grid[up[0]][up[1]] <= grid[curr_loc[0]][curr_loc[1]]:
                    index_list.append(up)
                if left not in index_list and left not in marked and left[1] >= 0 and \
                        grid[left[0]][left[1]] <= grid[curr_loc[0]][curr_loc[1]]:
                    index_list.append(left)

                down = [curr_loc[0] + 1, curr_loc[1]]
                right = [curr_loc[0], curr_loc[1] + 1]

                if down not in index_list and down not in marked and down[0] <= len(grid) - 1 and \
                        grid[down[0]][down[1]] <= grid[curr_loc[0]][curr_loc[1]]:
                    index_list.append(down)
                if right not in index_list and right not in marked and right[1] <= len(grid[0]) - 1 and \
                        grid[right[0]][right[1]] <= grid[curr_loc[0]][curr_loc[1]]:
                    index_list.append(right)
                if check_pacific and check_atlantic:
                    valid_list.append([row, col])
                    return True
                marked.append(index_list[0])
                index_list = index_list[1:]
            return False

        # def check_valid_atlantic(row, col, grid):
        #     index_list = [[row, col]]
        #     while len(index_list) > 0:
        #         curr_loc = index_list[0]
        #         if curr_loc[0] == len(grid) - 1 or curr_loc[1] == len(grid[0]) - 1:
        #             return True
        #         down = [curr_loc[0] + 1, curr_loc[1]]
        #         right = [curr_loc[0], curr_loc[1] + 1]
        #
        #         if down not in index_list and grid[down[0]][down[1]] <= grid[curr_loc[0]][curr_loc[1]]:
        #             index_list.append(down)
        #         if right not in index_list and grid[right[0]][right[1]] <= grid[curr_loc[0]][curr_loc[1]]:
        #             index_list.append(right)
        #         index_list = index_list[1:]
        #     return False

        res = []
        for i in range(len(heights)):
            for j in range(len(heights[0])):
                valid_list = []
                # pacific_check = check_valid_pacific(i, j, heights)
                # atlantic_check = check_valid_atlantic(i, j, heights)
                # if pacific_check and atlantic_check:
                #     res.append([i, j])
                if check_valid(i, j, heights, valid_list):
                    res.append([i, j])
        return res

    def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
        sum_alice = sum(aliceSizes)
        sum_bob = sum(bobSizes)
        average = (sum_alice + sum_bob) // 2
        for i in aliceSizes:
            if (sum_bob + 2 * i - sum_alice) // 2 in bobSizes:
                return [i, (sum_bob + 2 * i - sum_alice) // 2]

    def generateTheString(self, n: int) -> str:
        if n % 2 == 0:
            return 'x' * (n - 1) + 'a'
        return 'x' * n

    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        temp = [sum(x) for x in mat]
        order = sorted(range(len(temp)), key=lambda x: temp[x])
        return order[:k]

    def sortColors(self, nums: List[int]) -> None:
        bin = collections.defaultdict(int)
        for num in nums:
            bin[num] += 1
        idx = 0
        for item in sorted(bin.keys()):
            nums[idx:idx + bin[item]] = [item] * bin[item]
            idx += bin[item]

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        def merge_two(intv1, intv2):
            if intv1[0] <= intv2[0] and intv1[1] >= intv2[1]:
                return intv1
            if intv1[0] >= intv2[0] and intv1[1] <= intv2[1]:
                return intv2
            if intv1[0] <= intv2[0] <= intv1[1]:
                return [intv1[0], intv2[1]]
            elif intv1[0] <= intv2[1] <= intv1[1]:
                return [intv2[0], intv1[1]]
            else:
                return -1

        res = [intervals[0]]
        for i in range(1, len(intervals)):
            result = merge_two(res[-1], intervals[i])
            if result != -1:
                res[-1] = result
            else:
                res.append(intervals[i])
        if len(res) == 1:
            return res
        while True:
            res = sorted(res, key=lambda x: x[0])
            merge_flag = False
            temp_res = [res[0]]
            for i in range(1, len(res)):
                result = merge_two(temp_res[-1], res[i])
                if result != -1:
                    temp_res[-1] = result
                    merge_flag = True
                else:
                    temp_res.append(res[i])
            res = temp_res
            if not merge_flag:
                break
        return res

    def shuffle(self, nums: List[int], n: int) -> List[int]:
        res = [0] * (2 * n)
        for i in range(n):
            res[i * 2] = nums[i]
            res[i * 2 + 1] = nums[i + n]
        return res

    def reachNumber(self, target: int) -> int:
        # min = 0
        # max = 35000
        # while True:
        #     mid = (max - min) // 2
        #     if mid * (mid + 1) / 2 >= target > (mid - 1) * mid / 2:
        #         return mid
        #     elif (mid - 1) * mid / 2 >= target:
        #         max = mid
        #     elif mid * (mid + 1) / 2 < target:
        #         min = mid
        target = abs(target)
        k = 0
        while target > 0:
            k += 1
            target -= k

        if target % 2 == 0:
            return k
        return k + 1 + k % 2

    def isSameAfterReversals(self, num: int) -> bool:
        if num == 0:
            return True
        if num % 10 == 0:
            return False
        return True

    def validIPAddress(self, queryIP: str) -> str:
        def check_v4(ip):
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            for part in parts:
                if not part.isdigit():
                    return False
                if part[0] == '0' and part != '0':
                    return False
                if int(part) > 255 or int(part) < 0:
                    return False
            return True

        def check_v6(ip):
            parts = ip.split(':')
            if len(parts) != 8:
                return False
            for part in parts:
                if len(part) < 1 or len(part) > 4:
                    return False
                if not all(x in string.hexdigits for x in part):
                    return False
            return True

        v4, v6 = False, False
        if '.' in queryIP:
            v4 = check_v4(queryIP)
        elif ':' in queryIP:
            v6 = check_v6(queryIP)

        if v4:
            return 'IPv4'
        if v6:
            return 'IPv6'
        return 'Neither'

    def restoreIpAddresses(self, s: str) -> List[str]:
        def check_valid(x):
            if x[0] == '0' and x != '0':
                return False
            elif int(x) < 0 or int(x) > 255:
                return False
            return True

        # def backtracking(remain, cut_size, path, res):
        #     if cut_size == 3:
        #         return check_valid(remain)
        #     if cut_size >= 4:
        #         return False
        #     if remain == '':
        #         return cut_size == 5
        #     for i in range(1, len(remain)):
        #         cut_string = remain[:i]
        #         if not check_valid(cut_string):
        #             break
        #         path.append(cut_string)
        #         if backtracking(remain[i:], cut_size + 1, path, res):
        #             temp = [x for x in path]
        #             temp.append(remain[i:])
        #             res.append(temp)
        #             return True
        #         del path[-1]
        def backtracking(s, path, start, end):
            op = []
            if len(path) == 4 and start == end:
                return [path]
            for j in range(start + 1, end + 1):
                cut_string = s[start:j]
                if check_valid(cut_string):
                    temp = backtracking(s, path + [cut_string], j, end)
                    op.extend(temp)
            return op

        path = []
        res = []
        res = backtracking(s, path, 0, len(s))
        return ['.'.join(x) for x in res]

    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        def backtracking(nums, idx, curr_value, path, res):
            if idx > len(nums):
                return
            if len(path) >= 2:
                res.append(path)
            for i in range(idx, len(nums)):
                if nums[i] >= curr_value:
                    backtracking(nums, i + 1, nums[i], path + [nums[i]], res)

        res = []
        backtracking(nums, 0, -101, [], res)
        temp = []
        for item in res:
            if item not in temp:
                temp.append(item)
        return temp

    def longestNiceSubstring(self, s: str) -> str:
        def check_nice(x):
            for letter in x:
                if letter.swapcase() not in x:
                    return False
            return True

        res = []
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                if check_nice(s[i:j]):
                    res.append(s[i: j])
        res = sorted(res, key=lambda x: len(x), reverse=True)
        if len(res) == 0:
            return ''
        return res[0]

    def tribonacci(self, n: int) -> int:
        temp = [0] * (n + 1)
        temp[0:3] = [0, 1, 1]
        for i in range(3, len(temp)):
            temp[i] = temp[i - 1] + temp[i - 2] + temp[i - 3]
        return temp[n]

    def numSubmat(self, mat: List[List[int]]) -> int:
        m, n = len(mat), len(mat[0])
        nums = [[0] * n for i in mat]
        for i in range(m):
            nums[i][0] = mat[i][0]
            for j in range(1, n):
                if mat[i][j] == 1:
                    nums[i][j] = nums[i][j - 1] + 1
        res = 0
        for i in range(m):
            for j in range(n):
                temp = 1812
                for k in range(i, m):
                    temp = min(temp, nums[k][j])
                    res += temp

        return res

    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        count_diff = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                count_diff.append([s1[i], s2[i]])
        if count_diff == 0:
            return True
        if count_diff == 2:
            return count_diff[0][0] == count_diff[1][1] and count_diff[0][1] == count_diff[1][0]
        return False

    def search(self, nums: List[int], target: int) -> int:
        # def find_pivot(nums):
        #     if len(nums) == 1:
        #         return 0
        #     start, end = 0, len(nums)
        #     mid = 0
        #     while start < end:
        #         mid = (end + start) // 2
        #         if nums[mid - 1] > nums[mid] and nums[(mid + 1) % len(nums)] > nums[mid]:
        #             return mid
        #         if nums[mid] < nums[-1]:
        #             end = mid - 1
        #         else:
        #             start = mid + 1
        #     return 0
        #
        # def binary_search(nums, start, end, k):
        #     while start <= end:
        #         mid = (start + end) // 2
        #         if nums[mid] == k:
        #             return mid
        #         if nums[mid] > k:
        #             end = mid - 1
        #         else:
        #             start = mid + 1
        #     return -1
        #
        # pivot = find_pivot(nums)
        # if nums[pivot] == target:
        #     return pivot
        # if nums[0] <= target <= nums[pivot - 1]:
        #     return binary_search(nums, 0, pivot, target)
        # elif nums[pivot] <= target <= nums[-1]:
        #     return binary_search(nums, pivot, len(nums), target)
        # else:
        #     return -1

        start, end = 0, len(nums) - 1
        while start <= end:
            mid = (start + end) // 2
            if nums[mid] == target:
                return mid
            if nums[start] <= nums[mid]:
                if nums[start] > target or nums[mid] < target:
                    start = mid + 1
                else:
                    end = mid - 1
            else:
                if nums[mid] > target or nums[end] < target:
                    end = mid - 1
                else:
                    start = mid + 1
        return -1

    def getRow(self, rowIndex: int) -> List[int]:
        if rowIndex == 0 or rowIndex == 1:
            return [1] * (rowIndex + 1)
        last_row = [1, 1]
        for i in range(2, rowIndex + 1):
            new_row = [1]
            for k in range(len(last_row) - 1):
                new_row.append(last_row[k] + last_row[k + 1])
            new_row.append(1)
            last_row = new_row

        return last_row

    def search2(self, nums: List[int], target: int) -> bool:
        nums = list(dict.fromkeys(nums))
        start, end = 0, len(nums) - 1
        while start <= end:
            mid = (start + end) // 2
            if nums[mid] == target:
                return True
            if nums[start] <= nums[mid]:
                if nums[start] > target or nums[mid] < target:
                    start = mid + 1
                else:
                    end = mid - 1
            else:
                if nums[mid] > target or nums[end] < target:
                    end = mid - 1
                else:
                    start = mid + 1
        return False

    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix) - 1
        max_idx = len(matrix) - 1
        m = len(matrix) // 2 + len(matrix) % 2
        for i in range(m):
            for j in range(i, max_idx):
                a, b, c, d = matrix[i][j], matrix[j][n - i], matrix[n - i][n - j], matrix[n - j][i]
                matrix[i][j], matrix[j][n - i], matrix[n - i][n - j], matrix[n - j][i] = d, a, b, c
            max_idx -= 1

    def generateMatrix(self, n: int) -> List[List[int]]:
        def fill_with_dir(grid, row_idx, col_idx, dir, start_value, num):
            # end_idx = [row_idx + row_idx + dir[0] * num, col_idx + dir[1] * num]
            # grid[row_idx: max(end_idx[0], 1), col_idx:end_idx[1] + 1] = np.arange(start_value, start_value + num)
            end_idx = None
            if dir == [0, 1]:
                end_idx = [row_idx, col_idx + num]
                grid[row_idx, col_idx: end_idx[1]] = np.arange(start_value, start_value + num)
                return end_idx

            if dir == [1, 0]:
                end_idx = [row_idx + num, col_idx]
                grid[row_idx:end_idx[0], col_idx] = np.arange(start_value, start_value + num)
                return end_idx

            if dir == [0, -1]:
                end_idx = [row_idx, col_idx - num]
                grid[row_idx, end_idx[1] + 1: col_idx + 1] = np.arange(start_value + num - 1, start_value - 1, -1)
                return end_idx

            if dir == [-1, 0]:
                end_idx = [row_idx - num, col_idx]
                grid[end_idx[0] + 1:row_idx + 1, col_idx] = np.arange(start_value + num - 1, start_value - 1, -1)
                return end_idx

        num_layer = n // 2
        res = np.zeros((n, n))
        start_value = 1
        start_row, start_col = 0, 0
        start_num = n - 1
        dir = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        while num_layer > 0:
            count = 0
            while count < 4:
                end_idx = fill_with_dir(res, start_row, start_col, dir[count], start_value, start_num)
                start_row, start_col = end_idx[0], end_idx[1]
                count += 1
                start_value += start_num
                # if start_value > n ** 2:
                #     return res.astype(int).tolist()
            start_num -= 2
            start_row += 1
            start_col += 1
            num_layer -= 1
        print(start_value)
        if n % 2 == 1:
            res[n // 2, n // 2] = start_value
        return res.astype(int).tolist()

    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        def rotate(matrix):
            n = len(matrix)
            matrix = np.array(matrix)
            temp = np.zeros((n, n))
            for i in range(n):
                temp[:, n - 1 - i] = matrix[i, :]
            return temp

        if (np.array(mat) == np.array(target)).all():
            return True
        for i in range(3):
            mat = rotate(mat)
            if (np.array(mat) == np.array(target)).all():
                return True
        return False

    def countHomogenous(self, s: str) -> int:
        count = []
        start = 0
        end = 1
        while end < len(s):
            if s[end] == s[start]:
                end += 1
            else:
                count.append(end - start)
                start = end
                end += 1
        count.append(end - start)
        res = 0
        for num in count:
            res += math.comb(num + 1, 2)
        return res % (10 ** 9 + 7)

    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        for i in num:
            while stack and k > 0 and int(stack[-1]) > int(i):
                stack.pop(-1)
                k -= 1
            if i != '0' or stack:
                stack.append(i)

        if k > 0:
            stack = stack[:-k]
        if stack:
            return ''.join(stack)
        return '0'

    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        if n <= 2:
            if len(mines) == n ** 2:
                return 0
            else:
                return 1
        stack_down = [[1] * n for i in range(n)]
        stack_up = [[1] * n for i in range(n)]
        stack_left = [[1] * n for i in range(n)]
        stack_right = [[1] * n for i in range(n)]

        for mine in mines:
            i, j = mine[0], mine[1]
            stack_down[i][j] = 0
            stack_up[i][j] = 0
            stack_right[i][j] = 0
            stack_left[i][j] = 0

        for j in range(1, n - 1):
            for i in range(1, n - 1):
                if stack_down[i][j] != 0:
                    stack_down[i][j] = stack_down[i - 1][j] + 1

            for k in range(n - 2, 0, -1):
                if stack_up[k][j] != 0:
                    stack_up[k][j] = stack_up[k + 1][j] + 1

            for i in range(1, n - 1):
                if stack_right[j][i] != 0:
                    stack_right[j][i] = stack_right[j][i - 1] + 1

            for k in range(n - 2, 0, -1):
                if stack_left[j][k] != 0:
                    stack_left[j][k] = stack_left[j][k + 1] + 1
        res = 1
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                temp = min([stack_down[i][j], stack_left[i][j], stack_right[i][j], stack_up[i][j]])
                res = max(res, temp)
        return res

    def makeFancyString(self, s: str) -> str:
        res = ''
        start = 0
        end = 1
        while end < len(s):
            if s[end] == s[start]:
                end += 1
            else:
                res += s[start] * min(2, end - start)
                start = end
                end += 1
        res += s[start] * min(2, end - start)
        return res

    def findMaximumXOR(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            for j in range(i, n):
                res = max(res, nums[i] ^ nums[j])
        return res

    def sequentialDigits(self, low: int, high: int) -> List[int]:
        low_len = len(str(low))
        high_len = len(str(high))
        res = []
        for i in range(low_len, high_len + 1):
            for num in range(1, 10 - i + 1):
                temp = list(range(num, num + i))
                temp = [str(x) for x in temp]
                temp = int(''.join(temp))
                if low <= temp <= high:
                    res.append(temp)
                elif temp > high:
                    break

        return res

    def findDuplicate(self, nums: List[int]) -> int:
        # res = nums[0]
        # for idx, value in enumerate(nums[1:]):
        #     res ^= value ^ (idx + 1)
        # return res
        start, end = 0, len(nums) - 1
        res = 0
        while start <= end:
            mid = (start + end) // 2
            # count = 0

            count = sum(num <= mid for num in nums)
            if count > mid:
                res = mid
                end = mid - 1
            else:
                start = mid + 1
        return res

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        def divide(mat, k, up_left_corner, down_right_corner):
            # if len(mat) == 1 and len(mat[0] == 1):
            #     return mat[0][0] == k

            up_left_row, up_left_col = up_left_corner[0], up_left_corner[1]
            down_right_row, down_right_col = down_right_corner[0], down_right_corner[1]

            if up_left_row == down_right_row and up_left_col == down_right_col:
                return mat[up_left_row][up_left_col] == k
            mid_row = (up_left_row + down_right_row) // 2
            mid_col = (up_left_col + down_right_col) // 2
            mid_point = [mid_row, mid_col]

            up_left_check = False
            up_right_check = False
            down_left_check = False
            down_right_check = False

            if mat[up_left_row][up_left_col] <= k <= mat[mid_row][mid_col]:
                up_left_check = divide(mat, k, up_left_corner, mid_point)
                if up_left_check:
                    return True
            if (mid_col + 1 < len(mat[0])) and mat[up_left_row][mid_col + 1] <= k <= mat[mid_row][down_right_col]:
                up_right_check = divide(mat, k, [up_left_row, mid_col + 1], [mid_row, down_right_col])
                if up_right_check:
                    return True
            if (mid_row + 1 < len(mat)) and mat[mid_row + 1][up_left_col] <= k <= mat[down_right_row][mid_col]:
                down_left_check = divide(mat, k, [mid_row + 1, up_left_col], [down_right_row, mid_col])
                if down_left_check:
                    return True
            if (mid_row + 1 < len(mat)) and (mid_col + 1 < len(mat[0])) and \
                    mat[mid_row + 1][mid_col + 1] <= k <= mat[down_right_row][down_right_col]:
                down_right_check = divide(mat, k, [mid_row + 1, mid_col + 1], down_right_corner)
                if down_right_check:
                    return True

            return False

        return divide(matrix, target, [0, 0], [len(matrix) - 1, len(matrix[0]) - 1])

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals = sorted(intervals, key=lambda x: x[1])
        curr_end = intervals[0][1]
        res = 0
        for interval in intervals[1:]:
            if interval[0] < curr_end <= interval[1]:
                res += 1
            else:
                curr_end = interval[1]
        return res

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def calc_sum_hour(k, nums):
            return sum([math.ceil(num / k) for num in nums])

        res = 10 ** 9
        start = 0
        end = 10 ** 9
        while start <= end:
            mid = (start + end) // 2
            hours = calc_sum_hour(mid, piles)
            if hours <= h:
                res = min(res, mid)
                end = mid - 1
            else:
                start = mid + 1
        return res

    def checkPowersOfThree(self, n: int) -> bool:
        curr_max = 10 ** 9
        while n > 0:
            if n == 1:
                max_exp = 0
            else:
                max_exp = round(math.log(n, 3))
            remain = n - 3 ** max_exp
            if curr_max == max_exp:
                return False
            if remain == 0:
                return True
            n = remain

            curr_max = max_exp
        return False

    def secondHighest(self, s: str) -> int:
        first = -1
        second = -1
        for i in s:
            if i.isdigit():
                if int(i) > first:
                    if first != -1:
                        second = first
                    first = int(i)
                elif int(i) < first:
                    second = max(int(i), second)
        return second


def main():
    s = "abc1111"
    test = Solution()
    res = test.secondHighest(s)
    print(res)


if __name__ == '__main__':
    main()
