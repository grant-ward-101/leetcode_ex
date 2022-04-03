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

    def longestPalindrome(self, s: str) -> str:
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


def main():
    nums = [5]
    test = Solution()
    res = test.search(nums, 5)
    print(res)


if __name__ == '__main__':
    main()
