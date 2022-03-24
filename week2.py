import numpy as np
import bisect


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class MinStack:

    def __init__(self):
        self.stack = []
        self.min_val = None

    def push(self, val: int) -> None:
        self.stack.append(val)
        if self.min_val is None or self.min_val >= val:
            self.min_val = val

    def pop(self) -> None:
        temp = self.stack[-1]
        del self.stack[-1]
        if len(self.stack) > 0:
            self.min_val = min(self.stack)
        else:
            self.min_val = None
        return temp

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_val


class MyQueue:

    def __init__(self):
        self.queue = []

    def push(self, x: int) -> None:
        self.queue.append(x)

    def pop(self) -> int:
        if len(self.queue) > 0:
            temp = self.queue[0]
            del self.queue[0]
            return temp

    def peek(self) -> int:
        return self.queue[0]

    def empty(self) -> bool:
        return len(self.queue) == 0


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class SubrectangleQueries:

    def __init__(self, rectangle: list[list[int]]):
        self.matrix = rectangle

    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        for row_idx in range(row1, row2 + 1):
            for col_idx in range(col1, col2 + 1):
                self.matrix[row_idx][col_idx] = newValue

    def getValue(self, row: int, col: int) -> int:
        return self.matrix[row][col]


class Solution:
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        idx = len(matrix)
        for i in range(len(matrix)):
            if matrix[i][-1] >= target:
                idx = i
                break
        if idx == len(matrix):
            return False
        return target in matrix[idx]

    def isValidSudoku(self, board: list[list[str]]) -> bool:
        def check_3x3(arr):
            arr = np.ravel(arr)
            arr = np.delete(arr, np.where(arr == '.'))
            return len(arr) == len(set(arr))

        def check_line(arr):
            arr = np.delete(arr, np.where(arr == '.'))
            return len(arr) == len(set(arr))

        board = np.array(board)
        for i in range(0, len(board), 3):
            for j in range(0, len(board[i]), 3):
                small_board = board[i:i + 3, j:j + 3]
                if not check_3x3(small_board):
                    return False
        for i in range(len(board)):
            if not check_line(board[i]):
                return False
        for i in range(len(board[0])):
            if not check_line(board[:, i]):
                return False

        return True

    def firstUniqChar(self, s: str) -> int:
        hash_table = dict()
        for letter in s:
            if letter not in hash_table:
                hash_table[letter] = 1
            else:
                hash_table[letter] += 1
        for key in hash_table.keys():
            if hash_table[key] == 1:
                return s.find(key)
        return -1

    def isAnagram(self, s: str, t: str) -> bool:
        def create_hash_table(x):
            hash_table = dict()
            for letter in x:
                if letter not in hash_table:
                    hash_table[letter] = 1
                else:
                    hash_table[letter] += 1
            return hash_table

        s_hash_table = create_hash_table(s)
        t_hash_table = create_hash_table(t)
        return t_hash_table == s_hash_table

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        def create_hash_table(s):
            hash_table = dict()
            for letter in s:
                if letter not in hash_table:
                    hash_table[letter] = 1
                else:
                    hash_table[letter] += 1
            return hash_table

        note_hash = create_hash_table(ransomNote)
        mag_hash = create_hash_table(magazine)

        for key in note_hash:
            if key not in mag_hash:
                return False
            if note_hash[key] > mag_hash[key]:
                return False
        return True

    def mergeTwoLists(self, list1: [ListNode], list2: [ListNode]) -> [ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        if not list1 and not list2:
            return None

        if list1.val <= list2.val:
            head = list1
            list1 = list1.next
        else:
            head = list2
            list2 = list2.next

        temp = head
        while list1 and list2:
            if list1.val <= list2.val:
                temp.next = list1
                list1 = list1.next
            else:
                temp.next = list2
                list2 = list2.next
            temp = temp.next

        if list1:
            temp.next = list1
        else:
            temp.next = list2
        return head

    def removeElements(self, head: [ListNode], val: int) -> [ListNode]:
        if not head:
            return head
        temp = head
        while temp.next:
            if temp.next.val == val:
                temp.next = temp.next.next
            else:
                temp = temp.next
        if head.val == val:
            return head.next
        return head

    def hasCycle(self, head: [ListNode]) -> bool:
        seen = []
        temp = head
        while temp:
            if temp not in seen:
                seen.append(temp)
            else:
                return True
            temp = temp.next
        return False

    def deleteDuplicates(self, head: [ListNode]) -> [ListNode]:
        hash_table = dict()
        if not head:
            return None
        hash_table[head.val] = 1
        temp = head.next
        pre = head
        while temp:
            if temp.val in hash_table:
                pre.next = temp.next
            else:
                hash_table[temp.val] = 1
                pre = temp
            temp = temp.next
        return head

    def reverseList(self, head: [ListNode]) -> [ListNode]:
        if not head or not head.next:
            return head

        prev = None
        current = head

        while current:
            temp = current.next
            current.next = prev
            prev = current
            current = temp
        return prev

    def minDominoRotations(self, tops: list[int], bottoms: list[int]) -> int:
        def swap_one_face(first, second):
            min_rotation = 10 ** 5
            for value in range(1, 7):
                count = 0
                i = 0
                while i < len(first):
                    if first[i] == value:
                        i += 1
                        continue
                    elif second[i] == value:
                        i += 1
                        count += 1
                    else:
                        break

                if count < min_rotation and i == len(first):
                    min_rotation = count

            return -1 if min_rotation == 10 ** 5 else min_rotation

        top_swap = swap_one_face(tops, bottoms)
        bot_swap = swap_one_face(bottoms, tops)
        if top_swap == -1 or bot_swap == -1:
            return -1
        return min(top_swap, bot_swap)

    def isPowerOfTwo(self, n: int) -> bool:
        def recursive_divide(x):
            if x == 0:
                return False
            if x == 1:
                return True
            if x / 2 != x // 2:
                return False
            else:
                return recursive_divide(x // 2)

        return recursive_divide(n)

    def firstPalindrome(self, words: list[str]) -> str:
        def check_palindrome(x):
            return x == x[::-1]

        for word in words:
            if check_palindrome(word):
                return word
        return ''

    def isValid(self, s: str) -> bool:
        stack = []
        for letter in s:
            if letter in ['(', '{', '[']:
                stack.append(letter)
            else:
                if len(stack) > 0:
                    if (stack[-1] == '[' and letter == ']') or \
                            (stack[-1] == '{' and letter == '}') or \
                            (stack[-1] == '(' and letter == ')'):
                        del stack[-1]
                    else:
                        return False
                else:
                    return False
        if len(stack) > 0:
            return False
        return True

    def partitionLabels(self, s: str) -> list[int]:
        res = []
        current_idx = 0
        while len(s) > 0:
            current_letter = s[current_idx]
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

    def isSubtree(self, root: [TreeNode], subRoot: [TreeNode]) -> bool:
        def check_identical(node1, node2):
            if not node1 and not node2:
                return True
            if (not node1 and node2) or (not node2 and node1) or node1.val != node2.val:
                return False
            else:
                return check_identical(node1.left, node2.left) and check_identical(node1.right, node2.right)

        res = False

        def traverse(root, sub_root):
            nonlocal res
            if not root:
                return
            traverse(root.left, sub_root)
            if root.val == subRoot.val:
                if check_identical(root, sub_root):
                    res = True
            traverse(root.right, sub_root)

        traverse(root, subRoot)
        return res

    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
        year -= month < 3
        cal = year + int(year / 4) - int(year / 100) + int(year / 400) + t[month - 1] + day
        idx = cal % 7
        return days[idx - 1]

    def minCostClimbingStairs(self, cost: list[int]) -> int:
        a = cost[0]
        b = cost[1]

        for i in range(2, len(cost)):
            temp = cost[i] + min(a, b)
            a = b
            b = temp
        return min(a, b)

    def prefixCount(self, words: list[str], pref: str) -> int:
        count = 0
        for word in words:
            if word[:len(pref)] == pref:
                count += 1
        return count

    def countMatches(self, items: list[list[str]], ruleKey: str, ruleValue: str) -> int:
        def check_matching(values, rule_key, rule_value):
            if rule_key == 'type' and rule_value == values[0]:
                return True
            if rule_key == 'color' and rule_value == values[1]:
                return True
            if rule_key == 'name' and rule_value == values[2]:
                return True
            return False

        count = 0
        for item in items:
            if check_matching(item, ruleKey, ruleValue):
                count += 1
        return count

    def findPoisonedDuration(self, timeSeries: list[int], duration: int) -> int:
        total = 0
        for i in range(len(timeSeries) - 1):
            if timeSeries[i] + duration <= timeSeries[i + 1]:
                total += duration
            else:
                total += timeSeries[i + 1] - timeSeries[i]
        total += duration
        return total

    def countBinarySubstrings(self, s: str) -> int:
        prev, curr = 0, 1
        count = 0
        for i in range(1, len(s)):
            if s[i - 1] == s[i]:
                curr += 1
            else:
                count += min(prev, curr)
                prev = curr
                curr = 1
        count += min(prev, curr)
        return count

    def countPairs(self, nums: list[int], k: int) -> int:
        count = 0
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if nums[i] == nums[j] and i * j % k == 0:
                    count += 1
        return count

    def containsPattern(self, arr: list[int], m: int, k: int) -> bool:
        # for i in range(len(arr) - m):
        #     pattern = arr[i:i + m]
        #     j = i + m
        #     count = 0
        #     flag = 0
        #     while j < len(arr):
        #         if arr[j:j + m] == pattern:
        #             count += 1
        #             j += m
        #             if count >= k - 1 and flag == 0:
        #                 return True
        #         else:
        #             flag = 1
        #             j += 1
        # return False

        for i in range(len(arr) - m * k + 1):
            if arr[i:i + m] * k == arr[i:i + m * k]:
                return True
        return False

    def maximumPopulation(self, logs: list[list[int]]) -> int:
        population_dict = dict()
        min_year = min([min(x) for x in logs])
        max_year = max([max(x) for x in logs])

        for i in range(min_year, max_year):
            curr_pop = 0
            for log in logs:
                if log[0] <= i < log[1]:
                    curr_pop += 1
            population_dict[i] = curr_pop

        return max(population_dict, key=population_dict.get)

    def uniqueOccurrences(self, arr: list[int]) -> bool:
        occur_dict = dict()
        for i in arr:
            if i not in occur_dict:
                occur_dict[i] = 1
            else:
                occur_dict[i] += 1

        occurrences = occur_dict.values()
        return len(occurrences) == len(set(occurrences))

    def sumZero(self, n: int) -> list[int]:
        res = []
        if n % 2 == 0:
            temp = n
        else:
            temp = n - 1
            res.append(0)
        for i in range(1, int(temp / 2) + 1):
            res.append(i)
            res.append(-1 * i)
        return res

    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        res = 0
        full_bottles = numBottles
        empty_bottles = 0
        while full_bottles > 0:
            res += full_bottles
            empty_bottles = full_bottles + empty_bottles

            full_bottles = empty_bottles // numExchange
            empty_bottles = empty_bottles % numExchange
        return res

    def preorderTraversal(self, root: [TreeNode]) -> list[int]:
        def recursive(root, res):
            if not root:
                return res
            res.append(root.val)
            recursive(root.left, res)
            recursive(root.right, res)
            return res

        res = []
        return recursive(root, res)

    def inorderTraversal(self, root: [TreeNode]) -> list[int]:
        def recursive(root, res):
            if not root:
                return res
            recursive(root.left, res)
            res.append(root.val)
            recursive(root.right, res)
            return res

        res = []
        return recursive(root, res)

    def postorderTraversal(self, root: [TreeNode]) -> list[int]:
        def recursive(root, res):
            if not root:
                return res
            recursive(root.left, res)
            recursive(root.right, res)
            res.append(root.val)
            return res

        res = []
        return recursive(root, res)

    def getSmallestString(self, n: int, k: int) -> str:
        res = ['a'] * n
        # while idx < n:
        #     sum_diff = k - sum(res) + 1
        #     if sum_diff >= 26:
        #         num_rep = sum_diff // 26
        #         res[idx:idx + num_rep] = [26] * num_rep
        #         idx += num_rep
        #     else:
        #         res[idx] = sum_diff
        #         idx += 1
        sum_diff = k - n
        num_rep_z = sum_diff // 25
        filled_letter = sum_diff % 25
        res[:num_rep_z] = 'z' * num_rep_z
        if num_rep_z < n:
            res[num_rep_z] = chr(filled_letter + 97)

        res = res[::-1]
        return ''.join(res)

    def addTwoNumbers(self, l1: [ListNode], l2: [ListNode]) -> [ListNode]:
        res = ListNode()
        temp_res = res
        if not l1:
            return l2
        if not l2:
            return l1
        left_over = 0
        while l1 or l2:
            v1 = 0
            v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            temp = v1 + v2 + left_over
            temp_res.next = ListNode(temp % 10)
            left_over = temp // 10
            temp_res = temp_res.next
        if left_over > 0:
            temp_res.next = ListNode(left_over)
        return res.next

    def convertToBase7(self, num: int) -> str:
        res = ''
        negative = False
        if num == 0:
            return '0'
        if num < 0:
            num *= -1
            negative = True
        while num > 0:
            res += str(num % 7)
            num = num // 7
        res = res[::-1]
        if negative:
            res = '-' + res
        return res

    def heightChecker(self, heights: list[int]) -> int:
        sorted_height = sorted(heights)
        count = 0
        for i in range(len(heights)):
            if heights[i] != sorted_height[i]:
                count += 1
        return count

    def numSubseq(self, nums: list[int], target: int) -> int:
        res = 0
        max_num = 10 ** 9 + 7
        nums = sorted(nums)
        for i in range(len(nums)):
            loc = bisect.bisect_right(nums, target - nums[i])
            if loc > i:
                res += pow(2, loc - i - 1, max_num)
            else:
                break
        return res % max_num

    def maxMatrixSum(self, matrix: list[list[int]]) -> int:
        flatten = []
        number_negative = 0
        for row in matrix:
            for i in row:
                if i < 0:
                    flatten.append(i * -1)
                    number_negative += 1
                else:
                    flatten.append(i)
        if number_negative % 2 == 0:
            return sum(flatten)
        else:
            return sum(flatten) - 2 * min(flatten)

    def brokenCalc(self, startValue: int, target: int) -> int:
        def backward_cal(end_value, start_value):
            if start_value >= end_value:
                return start_value - end_value
            if end_value % 2 == 0:
                return 1 + backward_cal(end_value / 2, start_value)
            else:
                return 1 + backward_cal(end_value + 1, start_value)

        return int(backward_cal(target, startValue))

    def isSymmetric(self, root: [TreeNode]) -> bool:
        def check_mirror(node1, node2):
            if node1 is None and node2 is None:
                return True
            if (node1 is None and node2 is not None) or \
                    (node1 is not None and node2 is None) or \
                    (node1.val != node2.val):
                return False
            else:
                return check_mirror(node1.left, node2.right) & check_mirror(node1.right, node2.left)

        return check_mirror(root.left, root.right)

    def maxDepth(self, root: [TreeNode]) -> int:
        def traverse_to_leaf(node):
            if not node:
                return 0
            return 1 + max(traverse_to_leaf(node.left), traverse_to_leaf(node.right))

        return traverse_to_leaf(root)

    def levelOrder(self, root: [TreeNode]) -> list[list[int]]:
        res = []
        if not root:
            return res

        def get_layer_values(node, output, level=0):
            if not node:
                return output
            if level == len(output):
                output.append([node.val])
            else:
                output[level].append(node.val)
            get_layer_values(node.left, output, level + 1)
            get_layer_values(node.right, output, level + 1)

        get_layer_values(root, res, 0)
        return res

    def mostPoints(self, questions: list[list[int]]) -> int:
        temp = [0] * len(questions)
        temp[-1] = questions[-1][0]
        for i in range(len(questions) - 2, -1, -1):
            # if i + questions[i][1] < len(questions) - 1:
            #     temp[i] = max(questions[i][0] + temp[i + questions[i][1] + 1], questions[i + 1][0])
            # else:
            #     temp[i] = max(questions[i][0], temp[min(i + 1, len(questions) - 1)])
            temp_sum = questions[i][0]
            if i + questions[i][1] < len(questions) - 1:
                temp_sum += temp[i + questions[i][1] + 1]
            temp[i] = max(temp_sum, temp[i + 1])
        return max(temp)

    def maximumBinaryString(self, binary: str) -> str:
        pre_ones = 0
        for i in binary:
            if i == '1':
                pre_ones += 1
            else:
                break
        if pre_ones == len(binary):
            return binary
        remain = binary[pre_ones:]
        remain_zeros = 0
        for x in remain:
            if x == '0':
                remain_zeros += 1
        remain_ones = len(remain) - remain_zeros
        return '1' * (pre_ones + remain_zeros - 1) + '0' + '1' * remain_ones

    def countSegments(self, s: str) -> int:
        return len(s.split(''))

    def minFlips(self, s: str) -> int:
        # works but takes too long, probably due to looping

        # def count_pos(x):
        #     odd1 = 0
        #     even1 = 0
        #     odd0 = 0
        #     even0 = 0
        #     for i in range(len(x)):
        #         if i % 2 == 0 and x[i] == '0':
        #             even0 += 1
        #         if i % 2 == 0 and x[i] == '1':
        #             even1 += 1
        #         if i % 2 == 1 and x[i] == '0':
        #             odd0 += 1
        #         if i % 2 == 1 and x[i] == '1':
        #             odd1 += 1
        #     return (odd1, even1, odd0, even0)
        #
        # min_flip = 10 ** 5
        # for i in range(len(s)):
        #     ones_odd, ones_even, zeros_odd, zeros_even = count_pos(s)
        #     temp = min(ones_odd + zeros_even, ones_even + zeros_odd)
        #
        #     if min_flip > temp:
        #         min_flip = temp
        #     s = s[1:] + s[0]
        # return min_flip

        min_flips = 10 ** 5
        even_string = ''
        odd_string = ''
        double_string = s * 2
        for i in range(len(double_string)):
            if i % 2 == 0:
                even_string += '0'
                odd_string += '1'
            else:
                even_string += '1'
                odd_string += '0'
        even_count = 0
        odd_count = 0
        for i in range(len(double_string)):
            if double_string[i] != even_string[i]:
                even_count += 1
            if double_string[i] != odd_string[i]:
                odd_count += 1
            if i >= len(s):
                if double_string[i - len(s)] != even_string[i - len(s)]:
                    even_count -= 1
                if double_string[i - len(s)] != odd_string[i - len(s)]:
                    odd_count -= 1
            if i >= len(s) - 1:
                min_flips = min(min_flips, min(odd_count, even_count))
        return min_flips

    def checkString(self, s: str) -> bool:
        last_a_idx = s.rfind('a')
        first_b_idx = s.find('b')
        if last_a_idx == -1 or first_b_idx == -1:
            return True
        if last_a_idx < first_b_idx:
            return True
        return False

    def buddyStrings(self, s: str, goal: str) -> bool:
        if len(s) != len(goal):
            return False
        if s == goal:
            seen = dict()
            for i in s:
                if i not in seen:
                    seen[i] = 1
                else:
                    return True
            return False
        else:
            pairs = []
            for i in range(len(s)):
                if s[i] != goal[i]:
                    pairs.append([s[i], goal[i]])
                if len(pairs) > 2:
                    return False
            return len(pairs) == 2 and pairs[0] == pairs[1][::-1]

    def distributeCandies(self, candies: int, num_people: int) -> list[int]:
        remain = candies
        res = [0] * num_people
        required = 1
        while remain > 0:
            for i in range(num_people):
                if remain <= required:
                    res[i] += remain
                    return res
                else:
                    res[i] += required
                    remain -= required
                required += 1

    def invertTree(self, root: [TreeNode]) -> [TreeNode]:
        if not root:
            return None
        temp = root.left
        root.left = root.right
        root.right = temp
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

    def hasPathSum(self, root: [TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        if root.val == targetSum and root.left is None and root.right is None:
            return True
        else:
            return self.hasPathSum(root.left, targetSum - root.val) or \
                   self.hasPathSum(root.right, targetSum - root.val)


def main():
    candies = 10
    num_people = 3
    test = Solution()
    res = test.distributeCandies(candies, num_people)
    print(res)


if __name__ == '__main__':
    main()
