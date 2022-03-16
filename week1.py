from audioop import maxpp
from bisect import bisect
from pydoc import tempfilepager
import  sys
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        for idx in range(len(nums) - 1):
            subtract = target - nums[idx]
            if subtract in nums[idx + 1:]:
                return [idx, nums[idx + 1:].index(subtract) + idx + 1]


    def merge(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        # # def find_index(arr, x):
        # #     for i in range(len(arr)):
        # #         if arr[i] >= x:
        # #             return i
        # #     return n

        # from bisect import bisect
        # for i in nums2:
        #     idx = bisect(nums1, i)
            
        #     nums1 = nums1[:idx] + [i] + nums1[idx:-1]
        temp = nums1[:m]
        res = []
        i, j = 0, 0
        idx = 0
        while i < m and j < n:
            if temp[i] < nums2[j]:
                nums1[idx] = temp[i]
                i += 1
                idx += 1
            else:
                nums1[idx] = nums2[j]
                j += 1
                idx += 1
        # if i < m:
        #     nums1 += temp[i:]
        # else:
        #     nums1 += nums2[j:]
        while i < m:
            nums1[idx] = temp[i]
            i += 1
            idx += 1
        while j < n:
            nums1[idx] = nums2[j]
            j += 1
            idx += 1


    def maxSubsequence(self, nums: list[int], k: int) -> list[int]:
        # temp = dict(zip(nums, range(len(nums))))
        # print(temp)
        # res = []
        # sorted_nums = sorted(temp, reverse=True)
        # for key in sorted_nums:
        #     res.append(nums[temp[key]])
        # return res

        temp = enumerate(nums)
        reversed_temp = sorted(temp, reverse=True, key=lambda x: x[1])
        reversed_temp = [list(x) for x in reversed_temp][:k]
        
        reversed_temp = sorted(reversed_temp, key=lambda x: x[0])
        
        return [x[1] for x in reversed_temp]
        

    
    def reorderLogFiles(self, logs: list[str]) -> list[str]:
        let_log = []
        dig_log = []
        none_iden_log = []
        for i in logs:
            if i[-1].isdigit():
                if i[0].isdigit():
                    none_iden_log.append(i)
                else:
                    dig_log.append(i)
            else:
                let_log.append(i)
        
        sorted_let_log = sorted(let_log, key=lambda x: x.split(' ')[0])
        print(sorted_let_log)
        sorted_let_log = sorted(sorted_let_log, key=lambda x: x.split(' ')[1:])
        print(sorted_let_log)        
        # sorted_dig_log = sorted(dig_log, key=lambda x: x.split(' ')[0][-1])
        # print(dig_log)
        # print(sorted_dig_log)
        # sorted_none_iden_log = sorted(none_iden_log)
        # return sorted_let_log + sorted_dig_log + sorted_none_iden_log
        return sorted_let_log + dig_log



    def maxDepth(self, root: 'Node') -> int:

        def dfs(node, level):
            max_depth = 0
            if not node:
                return
            for child in node.children:
                max_depth = max(max_depth, dfs(child, level + 1))
            return max_depth
        
        return dfs(root, 1)


    def triangleNumber(self, nums: list[int]) -> int:
        res = 0
        nums.sort()
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                min_third_lenght = nums[i] + nums[j]
                third_location = bisect.bisect_left(nums, min_third_lenght)
                if third_location > j + 1:
                    res += third_location - j - 1
        return res


    def intersect(self, nums1: list[int], nums2: list[int]) -> list[int]:
        if len(nums1) > len(nums2):
            longer_list = nums1
            shorter_list = nums2
        else:
            longer_list = nums1
            shorter_list = nums2
        
        hash_map = dict()
        for i in longer_list:
            if i in hash_map:
                hash_map[i] += 1
            else:
                hash_map[i] = 1
        res = []
        for j in shorter_list:
            if j in hash_map and hash_map[j] > 0:
                res.append(j)
                hash_map[j] -= 1
        return res


    def maxProfit(self, prices: list[int]) -> int:
        max_profit = -1
        buy_day = 0
        for sell_date in range(1, len(prices)):
            current_profit = prices[sell_date] - prices[buy_day]
            if current_profit > max_profit:
                max_profit = current_profit
            if prices[sell_date] < prices[buy_day]:
                buy_day = sell_date
        
        return max_profit if max_profit > 0 else 0


    def matrixReshape(self, mat: list[list[int]], r: int, c: int) -> list[list[int]]:
        flatten_data = []
        for row in mat:
            flatten_data += row
        
        if r * c != len(flatten_data):
            return mat
        res = []
        while (len(flatten_data) > 0):
            res.append(flatten_data[:c])
            flatten_data = flatten_data[c:]
        return res


    def generate(self, numRows: int) -> list[list[int]]:
        
        def calculate_row(num):
            if num == 1:
                return [[1]]
            if num == 2:
                return [[1], [1, 1]]
            else:
                last_triangle = calculate_row(num - 1)
                last_row = last_triangle[-1]
                new_row = [1]
                for i in range(len(last_row) - 1):
                    new_row.append(last_row[i] + last_row[i + 1])
                new_row.append(1)
                last_triangle.append(new_row)
                return last_triangle
        return calculate_row(numRows)


    def validateStackSequences(self, pushed: list[int], popped: list[int]) -> bool:
        stack = []
        j = 0
        for i in pushed:
            stack.append(i)
            
            while (len(stack) > 0 and stack[-1] == popped[j]):
                stack.pop(len(stack) - 1)
                j += 1
        return len(stack) == 0 


    def replaceElements(self, arr: list[int]) -> list[int]:
        current_max = arr[-1]
        res = []
        for i in range(len(arr) - 2, -1, -1):
            temp = arr[i]
            if arr[i + 1] <= current_max:
                arr[i] = current_max
            if temp >= current_max:
                current_max = temp
        arr[-1] = -1
        return arr
            

def main():
    # s = sys.stdin.read()
    # print(s)
    # temp = s.split('\n')
    # k = int(temp[1])
    # nums = [int(x) for x in temp[0][1:-1].split(',')]

    s = [17,18,5,4,6,1]
    sol = Solution()
    res = sol.replaceElements(s)
    print(res)



if __name__=='__main__':
    main()
