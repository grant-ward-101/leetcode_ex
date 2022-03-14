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