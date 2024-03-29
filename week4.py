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


class MyLinkedList:
    class Node:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    def __init__(self):
        self.head = self.Node()
        self.length = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.length:
            return -1
        temp = self.head
        for i in range(index):
            temp = temp.next
        return temp.val

    def addAtHead(self, val: int) -> None:
        new_head = self.Node(val=val)
        if self.length > 0:
            new_head.next = self.head
        self.head = new_head
        self.length += 1

    def addAtTail(self, val: int) -> None:
        if self.length > 0:
            temp = self.head
            while temp and temp.next:
                temp = temp.next
            new_tail = self.Node(val=val)
            temp.next = new_tail
            self.length += 1
        else:
            self.addAtHead(val)

    def addAtIndex(self, index: int, val: int) -> None:
        if index == 0:
            self.addAtHead(val)
        elif index == self.length:
            self.addAtTail(val)
        elif 0 < index < self.length:
            new_node = self.Node(val=val)
            temp = self.head
            for i in range(index - 1):
                temp = temp.next
            next_node = temp.next
            temp.next = new_node
            new_node.next = next_node
            self.length += 1

    def deleteAtIndex(self, index: int) -> None:
        if self.length > 0:
            if index == 0:
                temp_head = self.head.next
                self.head.next = None
                self.head = temp_head
                self.length -= 1
            elif 0 < index < self.length:
                temp = self.head
                for i in range(index - 1):
                    temp = temp.next
                new_next_node = temp.next.next
                curr_next_node = temp.next
                temp.next = new_next_node
                curr_next_node.next = None
                self.length -= 1

    def print(self):
        s = []
        temp = self.head
        while temp:
            s.append(temp.val)
            temp = temp.next
        print(s)


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

    # def largestNumber(self, nums: List[int]) -> str:
    #     largest_num = ''.join(sorted(map(str, nums), key=larger_num))
    #     return '0' if largest_num[0] == '0' else largest_num

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

    def surfaceArea(self, grid: List[List[int]]) -> int:
        res = 0
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid)):
                if grid[i][j] > 0:
                    count += 1
                if i - 1 >= 0 and grid[i][j] > grid[i - 1][j]:
                    res += grid[i][j] - grid[i - 1][j]
                if i + 1 < len(grid) and grid[i][j] > grid[i + 1][j]:
                    res += grid[i][j] - grid[i + 1][j]
                if j - 1 >= 0 and grid[i][j] > grid[i][j - 1]:
                    res += grid[i][j] - grid[i][j - 1]
                if j + 1 < len(grid) and grid[i][j] > grid[i][j + 1]:
                    res += grid[i][j] - grid[i][j + 1]
        for i in range(len(grid)):
            res += grid[i][0]
            res += grid[0][i]
            res += grid[len(grid) - 1][i]
            res += grid[i][len(grid) - 1]
        return res + 2 * count

    def minDeletions(self, s: str) -> int:
        res = 0
        letter_counter = dict(collections.Counter(s).most_common())
        marked = []
        for letter, count in letter_counter.items():
            if count not in marked:
                marked.append(count)
            else:
                temp = letter_counter.values()
                while count in temp or count in marked:
                    res += 1
                    count -= 1
                if count != 0:
                    marked.append(count)
        return res

    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        # grid = np.array(grid)
        # for i in range(k):
        #     grid = np.roll(grid, 1, axis=1)
        #     grid[:, 0] = np.roll(grid[:, 0], 1, axis=0)
        # res = grid.tolist()
        # return res
        m = len(grid)
        n = len(grid[0])
        flatten = functools.reduce(operator.add, grid)
        k = k % len(flatten)
        new_grid = flatten[-k:] + flatten[:-k]
        reshape = [[new_grid[n * j + i] for i in range(n)] for j in range(m)]
        return reshape

    def numSub(self, s: str) -> int:
        start = 0
        end = 0
        temp = []
        for i in range(len(s)):
            if s[i] == '1':
                end += 1
            else:
                if s[i - 1] != '0':
                    temp.append(end - start)
                start = i + 1
                end = i + 1
        temp.append(end - start)
        res = 0
        mod = 10 ** 9 + 7
        for num in temp:
            res = (res + num * (num + 1) / 2) % mod
        return int(res)

    def removeOuterParentheses(self, s: str) -> str:
        stack = []
        res = ''
        start = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(s[i])
            else:
                stack.pop(-1)
            if len(stack) == 0:
                res += s[start + 1:i]
                start = i + 1
        return res

    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        time_dict = collections.defaultdict(int)
        releaseTimes = [0] + releaseTimes
        for idx, key in enumerate(keysPressed):
            time_dict[key] = max(time_dict[key], releaseTimes[idx + 1] - releaseTimes[idx])
        time_dict = dict(sorted(time_dict.items(), key=lambda x: x[0], reverse=True))
        time_dict = sorted(time_dict.items(), key=lambda x: x[1], reverse=True)
        return time_dict[0][0]

    def numMagicSquaresInside(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        grid = np.array(grid)

        def check_valid(mat):
            flatten = np.sort(mat, axis=None)
            if np.any(np.arange(1, 10, dtype=int) != flatten):
                return False
            sum_row = np.sum(mat, axis=1)
            if not np.all(sum_row == sum_row[0]):
                return False
            sum_col = np.sum(mat, axis=0)
            if not np.all(sum_col == sum_row[0]):
                return False
            first_dia = np.sum(mat.diagonal())
            second_dia = np.sum(np.fliplr(mat).diagonal())
            if first_dia != sum_row[0] or second_dia != sum_row[0]:
                return False
            return True

        res = 0
        for row in range(m - 2):
            for col in range(n - 2):
                temp = grid[row:row + 3, col:col + 3]
                if check_valid(temp):
                    res += 1
        return res
        # row_sum = np.zeros((m, n + 1))
        # col_sum = np.zeros((m + 1, n))

        # col_sum[1:, :] = np.cumsum(grid, axis=0)
        # row_sum[:, 1:] = np.cumsum(grid, axis=1)
        # res = 0
        # for row in range(m - 2):
        #     for col in range(n - 2):
        #         row_temp = row_sum[row:row + 3, col + 3] - row_sum[row:row + 3, col]
        #         if not np.all(row_temp == row_temp[0]):
        #             break
        #         col_temp = col_sum[row + 3, col: col + 3] - col_sum[row, col:col + 3]
        #         if not np.all(col_temp == row_temp[0]):
        #             break
        #         first_diagonal = grid[row][col] + grid[row + 1][col + 1] + grid[row + 2][col + 2]
        #         if first_diagonal != row_temp[0]:
        #             break
        #         second_diagonal = grid[row + 2][col] + grid[row + 1][col + 1] + grid[row][col + 2]
        #         if second_diagonal != row_temp[0]:
        #             break
        #         res += 1
        # return res

    def validPalindrome(self, s: str) -> bool:
        def check(low, high, mark=False):
            while low <= high:
                if s[low] != s[high]:
                    if mark:
                        return False
                    else:
                        return check(low + 1, high, True) or check(low, high - 1, True)
                else:
                    low += 1
                    high -= 1
            return True

        left = 0
        right = len(s) - 1

        return check(left, right)

    def isPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s) - 1
        while left <= right:
            if not s[left].isalnum():
                left += 1
                continue
            if not s[right].isalnum():
                right -= 1
                continue
            if s[left].lower() == s[right].lower():
                left += 1
                right -= 1
            else:
                return False
        return True

    def multiply(self, num1: str, num2: str) -> str:
        def sum_string(num1: str, num2: str) -> str:
            carry = 0
            res = ''
            while num1 and num2:
                sum = int(num1[-1]) + int(num2[-1]) + carry
                temp = sum % 10
                carry = sum // 10
                res = str(temp) + res
                num1 = num1[:-1]
                num2 = num2[:-1]
            while num1:
                sum = int(num1[-1]) + carry
                temp = sum % 10
                carry = sum // 10
                res = str(temp) + res
                num1 = num1[:-1]

            while num2:
                sum = int(num2[-1]) + carry
                temp = sum % 10
                carry = sum // 10
                res = str(temp) + res
                num2 = num2[:-1]
            if carry != 0:
                res = str(carry) + res
            return res

        def multiply_string(num1: str, num2: str) -> str:
            carry = 0
            res = ''
            while num1:
                mul = int(num1[-1]) * int(num2) + carry
                temp = mul % 10
                carry = mul // 10
                res = str(temp) + res
                num1 = num1[:-1]

            if carry != 0:
                res = str(carry) + res
            return res

        if num1 == '0' or num2 == '0':
            return '0'
        res = ''
        for i in range(len(num2) - 1, -1, -1):
            temp_mul = multiply_string(num1, num2[i])
            temp_mul += '0' * (len(num2) - 1 - i)
            res = sum_string(res, temp_mul)

        return res

    def shipWithinDays(self, weights: List[int], days: int) -> int:
        def calc_day(weights, load):
            count = 0
            temp_sum = 0
            if load < max(weights):
                return float('inf')
            for weight in weights:
                temp_sum += weight
                if temp_sum > load:
                    count += 1
                    temp_sum = weight
            return count + 1

        left = 0
        right = sum(weights)
        mid = 0
        while left <= right:
            mid = (left + right) // 2
            day = calc_day(weights, mid)
            if day <= days:
                right = mid - 1
            else:
                left = mid + 1
        return left

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        letter_dict = collections.defaultdict(list)
        for word in strs:
            sorted_word = ''.join(sorted(word))
            letter_dict[sorted_word].append(word)

        res = []
        for key, value in letter_dict.items():
            res.append(value)
        return res

    def nextPermutation(self, nums: List[int]) -> None:
        idx = len(nums) - 2
        while idx >= 0 and nums[idx] >= nums[idx + 1]:
            idx -= 1
        if idx >= 0:
            i = len(nums) - 1
            while nums[i] <= nums[idx]:
                i -= 1
            temp = nums[i]
            nums[i] = nums[idx]
            nums[idx] = temp
        idx += 1
        n = len(nums) - 1
        while idx < n:
            temp = nums[n]
            nums[n] = nums[idx]
            nums[idx] = temp
            idx += 1
            n -= 1

    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        dna_map = collections.defaultdict(int)
        res = []
        if len(s) <= 10:
            return res
        for i in range(len(s) - 9):
            temp = s[i:i + 10]
            dna_map[temp] += 1
        for key, value in dna_map.items():
            if value > 1:
                res.append(key)
        return res

    def longestPalindrome1(self, s: str) -> str:
        if len(s) <= 1:
            return s
        n = len(s)
        start = 0
        max_length = 0
        for i in range(1, n):
            left = i - 1
            right = i
            while left >= 0 and right <= n - 1 and s[left] == s[right]:
                left -= 1
                right += 1
            if right - left - 1 > max_length:
                start = left + 1
                max_length = right - left - 1

            left = i - 1
            right = i + 1
            while left >= 0 and right <= n - 1 and s[left] == s[right]:
                left -= 1
                right += 1
            if right - left - 1 > max_length:
                start = left + 1
                max_length = right - left - 1
        return s[start: start + max_length]

    def guessNumber(self, n: int) -> int:
        def guess(number):
            return 1

        left = 1
        right = n
        while left < right:
            mid = (left + right) // 2
            if guess(mid) == 0:
                return mid
            if guess(mid) == -1:
                left = mid + 1
            else:
                right = mid - 1

    def search(self, nums: list[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return -1

    def firstBadVersion(self, n: int) -> int:
        def isBadVersion(num):
            return True

        left = 0
        right = n
        while left <= right:
            mid = left + (right - left) // 2
            temp = isBadVersion(mid)
            if temp:
                right = mid - 1
            else:
                left = mid + 1
        return left

    def searchInsert(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return left

    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points = sorted(points, key=lambda x: x[1])

        idx = 0
        while idx < len(points) - 1:
            ball1 = points[idx]
            ball2 = points[idx + 1]
            flag = False

            if ball2[0] < ball1[0]:
                del points[idx + 1]
                flag = True
            elif ball2[0] <= ball1[1]:
                points[idx + 1] = [ball2[0], ball1[1]]
                del points[idx]
                flag = True

            if not flag:
                idx += 1

        return len(points)

    def rotate(self, nums: List[int], k: int) -> None:
        k = k % len(nums)

        for i in range(0, len(nums) // 2):
            nums[i], nums[len(nums) - 1 - i] = nums[len(nums) - 1 - i], nums[i]
        left = 0
        right = k - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

        left = k
        right = len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    def sortedSquares(self, nums: List[int]) -> List[int]:
        left = 0
        right = len(nums) - 1
        temp = []
        while left <= right:
            if abs(nums[left]) < abs(nums[right]):
                temp.append(abs(nums[right]))
                right -= 1
            else:
                temp.append(abs(nums[left]))
                left += 1
        res = [x ** 2 for x in temp[::-1]]
        return res

    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        left = 0
        right = len(arr) - 1
        while left < right:
            mid = left + (right - left) // 2
            if arr[mid] < arr[mid + 1]:
                left = mid + 1
            else:
                right = mid
        return left

    def swapNodes(self, head: [ListNode], k: int) -> [ListNode]:
        # left = head
        # right = head
        # prev_left = None
        # prev_right = None
        # for i in range(k - 1):
        #     prev_left, left = left, left.next
        #
        # temp = left
        # while temp.next:
        #     prev_right, right = right, right.next
        #     temp = temp.next
        # # temp_next_right = right.next
        # # temp_next_left = left.next
        # # prev_left.next = right
        # # right.next = temp_next_left
        # # prev_left.next = left
        # # left.next = temp_next_right
        # # if prev_left:
        # #     prev_left.next = right
        # # prev_right.next, left.next, right.next = left, right.next, left.next
        # return head
        left = head
        right = head
        for i in range(k - 1):
            left = left.next
        temp = left
        while temp.next:
            right = right.next
            temp = temp.next
        temp_val = left.val
        left.val = right.val
        right.val = temp_val
        return head

    def integerReplacement(self, n: int) -> int:
        def recursion(num):
            if num == 1:
                return 0
            else:
                if num % 2 == 0:
                    return 1 + recursion(num // 2)
                else:
                    return 1 + min(recursion(num + 1), recursion(num - 1))

        return recursion(n)

    def detectCycle(self, head: [ListNode]) -> [ListNode]:
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
            else:
                return None
        temp = head
        while temp != slow:
            temp, slow = temp.next, slow.next
        return temp

    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        temp = np.array(mat)
        m = len(mat)
        n = len(mat[0])
        prefix_sum = np.zeros((m + 1, n + 1))

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                prefix_sum[i, j] = prefix_sum[i - 1, j] \
                                   + prefix_sum[i, j - 1] \
                                   + temp[i - 1, j - 1] \
                                   - prefix_sum[i - 1, j - 1]

        res = np.zeros((m, n))
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                bot_right = prefix_sum[min(m, i + k), min(n, j + k)]
                bot_left = prefix_sum[min(m, i + k), max(0, j - k - 1)]
                top_right = prefix_sum[max(0, i - k - 1), min(n, j + k)]
                top_left = prefix_sum[max(0, i - k - 1), max(0, j - k - 1)]
                res[i - 1, j - 1] = bot_right - bot_left - top_right + top_left
        return res.tolist()

    def countOdds(self, low: int, high: int) -> int:
        subtract = high - low + 1
        if subtract % 2 and high % 2 and low % 2:
            return subtract // 2 + 1
        return subtract // 2

    def average(self, salary: List[int]) -> float:
        temp_min = salary[0]
        temp_max = salary[0]
        temp = salary[0] * -1
        for i in range(1, len(salary)):
            if salary[i] > temp_max:
                temp += temp_max
                temp_max = salary[i]
            elif salary[i] < temp_min:
                temp += temp_min
                temp_min = salary[i]
            else:
                temp += salary[i]
        return temp / (len(salary) - 2)

    def tree2str(self, root: [TreeNode]) -> str:
        def inorder(node):
            if not node:
                return ''
            if not node.left and not node.right:
                return str(node.val)

            temp = str(node.val)
            if not node.left and node.right:
                temp += '()'
            if node.left:
                temp += '(' + inorder(node.left) + ')'
            if node.right:
                temp += '(' + inorder(node.right) + ')'
            return temp

        if not root:
            return ''
        res = inorder(root)
        return res
        # res = res[1:-1]
        # stack = []
        # for letter in res:
        #     if stack and stack[-1] == '(' and letter == ')':
        #         stack = stack[:-1]
        #     else:
        #         stack.append(letter)
        # res = ''.join(stack)
        # return res

    def minOperations(self, grid: List[List[int]], x: int) -> int:
        mat = np.array(grid)
        if np.any(mat % x != mat[0][0] % x):
            return -1
        flatten = np.sort(np.ravel(mat))
        target = flatten[len(flatten) // 2]
        res = 0
        for num in flatten:
            res += abs(num - target) / x
        return int(res)

    # def maxArea(self, height: List[int]) -> int:
    #     left = 0
    #     right = len(height) - 1
    #     max_water = min(height[left], height[right]) * (right - left)
    #     while left < right:
    #         if right - left == 1:
    #             max_water = max(max_water, min(height[left], height[right]))
    #             break
    #         temp_water_left = 0
    #         greater_left = left
    #         while greater_left < right:
    #             if height[greater_left] > height[left]:
    #                 temp_water_left = min(height[greater_left], height[right]) * (right - greater_left)
    #                 break
    #             greater_left += 1
    #         if greater_left == right:
    #             greater_left = left
    #         temp_water_right = 0
    #         greater_right = right
    #         while greater_right > left:
    #             if height[greater_right] > height[right]:
    #                 temp_water_right = min(height[greater_right], height[left]) * (greater_right - left)
    #                 break
    #             greater_right -= 1
    #         if greater_right == left:
    #             greater_right = right
    #         if temp_water_left > temp_water_right:
    #             max_water = max(max_water, temp_water_left)
    #             left = greater_left
    #         else:
    #             max_water = max(max_water, temp_water_left)
    #             right = greater_right
    #     return max_water
    def maxArea(self, height: List[int]) -> int:
        max_water = 0
        left = 0
        right = len(height) - 1
        while left < right:
            max_water = max(max_water, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_water

    def isPerfectSquare(self, num: int) -> bool:
        if num == 1:
            return True
        left = 1
        right = num // 2
        while left <= right:
            mid = left + (right - left) // 2
            temp = mid * mid
            if temp == num:
                return True
            elif temp < num:
                left = mid + 1
            else:
                right = mid - 1
        return False

    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:

        res = 0
        arr2 = sorted(arr2)

        def binary_search(nums, target):
            low = 0
            high = len(nums) - 1
            while low <= high:
                mid = low + (high - low) // 2
                if abs(nums[mid] - target) <= d:
                    return False
                elif nums[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1
            return True

        for num1 in arr1:
            if binary_search(arr2, num1):
                res += 1
        return res

    def moveZeroes(self, nums: List[int]) -> None:
        count_zeros = 0
        idx = 0
        while idx < len(nums):
            if nums[idx] == 0:
                count_zeros += 1
                del nums[idx]
            else:
                idx += 1
        nums += [0] * count_zeros

    def subtractProductAndSum(self, n: int) -> int:
        product = 1
        sum = 0
        while n > 0:
            temp = n % 10
            product *= temp
            sum += temp
            n //= 10
        return product - sum

    def hammingWeight(self, n: int) -> int:
        count = 0
        while n > 0:
            if n % 2 == 0:
                count += 1
            n = n >> 1
        return count

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> [ListNode]:
        len_a, len_b = 0, 0
        temp_a, temp_b = headA, headB
        while temp_a:
            len_a += 1
            temp_a = temp_a.next
        while temp_b:
            len_b += 1
            temp_b = temp_b.next
        temp_a, temp_b = headA, headB
        for i in range(abs(len_a - len_b)):
            if len_a > len_b:
                temp_a = temp_a.next
            else:
                temp_b = temp_b.next
        while temp_a and temp_b and temp_a is not temp_b:
            temp_a = temp_a.next
            temp_b = temp_b.next
        return temp_a

    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1
        while left < right:
            temp = numbers[left] + numbers[right]
            if temp == target:
                return [left + 1, right + 1]
            elif temp < target:
                left += 1
            else:
                right -= 1

    def deleteDuplicates(self, head: [ListNode]) -> [ListNode]:
        seen = []
        duplicate = []
        temp = head
        while temp:
            if temp.val in seen:
                duplicate.append(temp.val)
            else:
                seen.append(temp.val)
            temp = temp.next
        while head and head.val in duplicate:
            head = head.next

        pre = None
        temp = head
        while temp:
            if temp.val not in duplicate:
                pre, temp = temp, temp.next
            else:
                next_not_duplicate = temp
                while next_not_duplicate and next_not_duplicate.val in duplicate:
                    next_not_duplicate = next_not_duplicate.next
                temp = next_not_duplicate
                pre.next = temp
        return head

    def maximumBobPoints(self, numArrows: int, aliceArrows: List[int]) -> List[int]:
        ans = []
        target = 0

        def rec(n, num_arrows, alice_arrows, sum, res):
            nonlocal ans, target
            if n == -1 or num_arrows <= 0:
                if sum > target:
                    target = sum
                    if num_arrows > 0:
                        res[0] += num_arrows
                    ans = res
                return
            required = alice_arrows[n] + 1
            if required <= num_arrows:
                res[n] = required
                rec(n - 1, num_arrows - required, alice_arrows, sum + n, res)
                res[n] = 0
            rec(n - 1, num_arrows, alice_arrows, sum, res)
            return

        res = [0] * 12
        rec(11, numArrows, aliceArrows, 0, res)
        return ans

    def longestPalindrome(self, words: List[str]) -> int:
        hash_map = collections.defaultdict(int)
        double = collections.defaultdict(int)
        for word in words:
            if word[0] != word[1]:
                hash_map[word] += 1
            else:
                double[word] += 1
        res = 0
        single = False
        seen = []
        for word in set(words):
            if word[0] != word[1]:
                res += min(hash_map[word], hash_map[word[::-1]])
            else:
                if double[word] % 2 == 0:
                    res += double[word]
                else:
                    if not single:
                        res += double[word]
                        single = True
                    else:
                        res += double[word] // 2 * 2
        return res * 2

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums = sorted(nums)
        res = float('inf')

        for i in range(len(nums) - 2):
            left = i + 1
            right = len(nums) - 1
            while left < right:
                temp_sum = nums[i] + nums[left] + nums[right]
                if abs(target - temp_sum) < abs(target - res):
                    res = temp_sum
                if temp_sum == target:
                    return temp_sum
                elif temp_sum > target:
                    right -= 1
                else:
                    left += 1
        return res

    def mySqrt(self, x: int) -> int:
        if x == 1:
            return x
        left = 0
        right = x // 2
        while left <= right:
            mid = left + (right - left) // 2
            if mid * mid == x:
                return mid
            elif mid * mid < x:
                left = mid + 1
            else:
                right = mid - 1
        return left - 1

    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        lower_target = ord(target) - 26
        upper_target = ord(target)

        letters = [ord(x) for x in letters]
        temp = 0
        if all(num <= upper_target for num in letters):
            temp = lower_target
        else:
            temp = upper_target

        left = 0
        right = len(letters) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if letters[mid] <= temp:
                left += 1
            else:
                right -= 1
        return chr(letters[left])

    def reverseWords(self, s: str) -> str:
        word_list = s.split()
        for idx, word in enumerate(word_list):
            word_list[idx] = word_list[idx][::-1]
        return ' '.join(word_list)

    def swapPairs(self, head: [ListNode]) -> [ListNode]:
        if not head or not head.next:
            return head
        dummy = ListNode()
        dummy.next = head
        node = head
        prev = dummy
        while node and node.next:
            next_two_temp = node.next.next
            second = node.next

            second.next = node
            node.next = next_two_temp
            prev.next = second

            prev = node
            node = next_two_temp
        return dummy.next


def main():
    # s = "Let's take LeetCode contest"
    # test = Solution()
    # res = test.reverseWords(s)
    # print(res)
    test = MyLinkedList()
    test.addAtHead(5)
    test.addAtIndex(1, 2)
    test.addAtHead(6)
    test.addAtTail(2)
    test.addAtTail(1)
    print(test.length)
    print(test.get(5))
    test.print()


if __name__ == '__main__':
    main()
