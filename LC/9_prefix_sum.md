title: "Presum"
date: 2020-03-28
tags: [[单调栈Monotone Stack](https://github.com/labuladong/fucking-algorithm/blob/master/数据结构系列/单调栈.md)],  [LC739.Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)]

206. Reverse Linked List
https://leetcode.com/problems/reverse-linked-list/



### 1，返回最近最大的值
---
<!--input：nums =  [2,1,2,4,3]
	output： res = [4,2,4,-1,-1]-->

### 分析
- 栈
- res [] ; len(res) = len(nums)
- 从后往前
![monotone_stack](pic/monotone_stack.png)

```python
def next_greater_element(T: list) -> list:
    res = [0 for _ in nums]
    stack = []
    for i in range(len(nums)-1,-1, -1):
        # stack is not empty
        while stack and nums[i] >= stack[-1]:
            stack.pop() 
        #stack is empty    
        if not stack:
            res[i] = -1
        else:
            res[i] = stack[-1] 
        stack.append(nums[i])
    return res
```

### 2，返回最近最大值的index

---
-  stack存的是index，比较要nums[index]
```python
def next_greater_element_index(T: list) -> list:
    res = [0 for _ in nums]
    stack = []
    for i in range(len(nums)-1,-1, -1):
        # stack is not empty
        # ！stack存的是index，比较要nums[index]
        while stack and nums[i] >= nums[stack[-1]]:
            stack.pop() 
        #stack is empty    
        if not stack:
            res[i] = -1
        else:
            res[i] = stack[-1] - i
        stack.append(i)
    return res
```
---


### [图片来源](https://github.com/labuladong/fucking-algorithm/blob/master/数据结构系列/单调栈.md)

