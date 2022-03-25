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
