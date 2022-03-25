import copy


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


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


def main():
    isWater = [[0, 1], [0, 0]]
    test = Solution()
    res = test.highestPeak(isWater)
    print(res)


if __name__ == '__main__':
    main()
