from bisect import bisect
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




def main():
    # s = sys.stdin.read()
    # print(s)
    # temp = s.split('\n')
    # k = int(temp[1])
    # nums = [int(x) for x in temp[0][1:-1].split(',')]
    s = [2, 2, 3, 4]
    sol = Solution()
    res = sol.triangleNumber(s)
    print(res)



if __name__=='__main__':
    main()
