# LeetCode Problem: Two Sum
#
# Given an array of integers `nums` and an integer `target`, return *indices of the two numbers* such that they add up to `target`.
#
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# You can return the answer in any order.
#
# **Example 1:**
# ```
# Input: nums = [2,7,11,15], target = 9
# Output: [0,1]
# Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
# ```
#
# **Example 2:**
# ```
# Input: nums = [3,2,4], target = 6
# Output: [1,2]
# ```
#
# **Example 3:**
# ```
# Input: nums = [3,3], target = 6
# Output: [0,1]
# ```

# ----------------------------------------------------------------

def two_sum(nums, target):
    """
    Returns the indices of the two numbers such that they add up to `target`.
    Uses a hash map for O(n) time complexity and O(n) space.

    :param list[int] nums: List of integers.
    :param int target: Target sum.
    :return tuple[int, int]: Indices of the two numbers.
    """
    num_to_index = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return (num_to_index[complement], i)
        num_to_index[num] = i
    raise ValueError("No two sum solution")

# ----------------------------------------------------------------
# Simple demonstration (can be removed or commented out)
if __name__ == "__main__":
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    print(f"Indices: {result}")  # Expected: (0, 1)