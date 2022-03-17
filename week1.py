from audioop import maxpp
from bisect import bisect
from cgitb import small
from pydoc import tempfilepager
import string
import  sys
from tabnanny import check
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


    def floodFill(self, image: list[list[int]], sr: int, sc: int, newColor: int) -> list[list[int]]: 
        marked = []
        def change_color(old, new, r, c, img):
            img[r][c] = new
            marked.append([r, c])
            if r - 1 >= 0 and img[r - 1][c] == old and [r - 1, c] not in marked:
                change_color(old, new, r - 1, c, img)
            if r + 1 <= len(img) - 1 and img[r + 1][c] == old and [r + 1, c] not in marked:
                change_color(old, new, r + 1, c, img)
            if c - 1 >= 0 and img[r][c - 1] == old and [r, c - 1] not in marked:
                change_color(old, new, r, c - 1, img)
            if c + 1 <= len(img[0]) - 1 and img[r][c + 1] == old and [r, c + 1] not in marked:
                change_color(old, new, r, c + 1, img)
        old = image[sr][sc]
        change_color(old, newColor, sr, sc, image)
        return image


    def uniqueMorseRepresentations(self, words: list[str]) -> int:
        arr = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        alphabets = list(string.ascii_lowercase)
        morse_dict = dict(zip(alphabets, arr))
        morsed_words = []
        for word in words:
            code = [morse_dict[x] for x in word]
            code = "".join(code)
            morsed_words.append(code)
        return len(set(morsed_words))


    def canBeEqual(self, target: list[int], arr: list[int]) -> bool:
        target_counter = collections.Counter(target)
        arr_counter = collections.Counter(arr)
        return target_counter == arr_counter


    def scoreOfParentheses(self, s: str) -> int: 
        score = 0
        stack = []
        for i in s:
            if i == '(':
                stack.append(score)
                score = 0
            else:
                score = stack[-1] + max(2 * score, 1)
                del stack[-1]
        return score


    def isValidSudoku(self, board: list[list[str]]) -> bool:
        import numpy as np 
        def check_duplicate(arr):
            arr.remove('.')
            return len(arr) == len(set(arr))

        def check_small_square(arr):
            flatten = []
            for i in range(len(arr)):
                for j in range(len(arr[0])):
                    flatten.append(arr[i][j])
            return check_duplicate(flatten)

        np_board = np.array(board)
        for i in range(0, len(np_board), 3):
            for j in range(0, len(np_board[0]), 3):
                small_board = np_board[i:i + 2, j:j + 2]
                if not check_small_square(small_board):
                    return False
        for i in range(len(np_board)):
            if not check_duplicate(np_board[i,:]):
                return False
        for i in range(len(np_board[0])):
            if not check_duplicate(np_board[:, i]):
                return False
        return True


class MyHashMap:
    
    def __init__(self):
        self.hash_map = []

    def put(self, key: int, value: int) -> None:
        key_list = [x[0] for x in self.hash_map]
        if key not in key_list:
            self.hash_map.append([key, value])
        else:
            pos = self.hash_map[:, 0].index(key)
            self.hash_map[pos, 1] = value

    def get(self, key: int) -> int:
        key_list = [x[0] for x in self.hash_map]
        if key not in key_list:
            return -1
        else:
            pos = key_list.index(key)
            return self.hash_map[pos, 1]

    def remove(self, key: int) -> None:
        key_list = [x[0] for x in self.hash_map]
        if key not in key_list:
            return
        else:
            pos = key_list.index(key)
            del self.hash_map[pos]
        
def main():
    # s = sys.stdin.read()
    # print(s)
    # temp = s.split('\n')
    # k = int(temp[1])
    # nums = [int(x) for x in temp[0][1:-1].split(',')]
    k = [[1, 1], [2, 2][2, 1]]
    print(k[:, 0])
    input()
    board = [["5","3",".",".","7",".",".",".","."]
            ,["6",".",".","1","9","5",".",".","."]
            ,[".","9","8",".",".",".",".","6","."]
            ,["8",".",".",".","6",".",".",".","3"]
            ,["4",".",".","8",".","3",".",".","1"]
            ,["7",".",".",".","2",".",".",".","6"]
            ,[".","6",".",".",".",".","2","8","."]
            ,[".",".",".","4","1","9",".",".","5"]
            ,[".",".",".",".","8",".",".","7","9"]]
    s = ["gin","zen","gig","msg"]
    sol = Solution()
    res = sol.isValidSudoku(board)
    print(res)



if __name__=='__main__':
    main()
