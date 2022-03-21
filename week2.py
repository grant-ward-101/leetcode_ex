import numpy as np


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
        if self.min_val == None or self.min_val >= val:
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
        else:
            return None

    def peek(self) -> int:
        return self.queue[0]

    def empty(self) -> bool:
        return len(self.queue) == 0


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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
        def create_hash_table(s):
            hash_table = dict()
            for letter in s:
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
        def recusive_divide(x):
            if x == 0:
                return False
            if x == 1:
                return True
            if x / 2 != x // 2:
                return False
            else:
                return recusive_divide(x // 2)

        return recusive_divide(n)

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
        def check_matching(item, rule_key, rule_value):
            if rule_key == 'type' and rule_value == item[0]:
                return True
            if rule_key == 'color' and rule_value == item[1]:
                return True
            if rule_key == 'name' and rule_value == item[2]:
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


def main():
    cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
    test = Solution()
    res = test.minCostClimbingStairs(cost)
    print(res)


if __name__ == '__main__':
    main()
