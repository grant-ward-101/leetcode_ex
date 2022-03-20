import numpy as np


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


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


def main():
    s = "anagram"
    t = "nagaram"
    s = Solution()
    res = s.isAnagram(s, t)
    print(res)


if __name__ == '__main__':
    main()
