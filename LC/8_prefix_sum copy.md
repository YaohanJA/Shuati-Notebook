title: "Presum"
date: 2020-03-28


### 1，返回最近最大的值
---


```python
'''
k 种颜色的颜料，每种颜料有 c[i] 升,
样例数目： 1
m,n,k : 1 5 2
颜料数量C[i]： 2 3
'''

def main():
    sample= int(input())

    for i in range(sample):
        m, n, k = map(int, input().split())
        num = (m * n + 1) // 2
        ci = map(int, input().split())

        flag = 0
        for i in ci:
            print(i)
            if i > num:
                flag = 1
                print("NO")
                break
        if not flag:
            print("YES")

main()
```

