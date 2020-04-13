title: "Presum"
date: 2020-04-09

### 1，countPrime
---
204. Count Primes
https://leetcode.com/problems/count-primes/submissions/
```python
class Solution:
    def countPrimes(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 0
        else: 
            isprime = [True for i in range(n)]
            isprime[0] = False
            isprime[1] = False
            for i in range(2, int(n**0.5)+1):
                if isprime[i]:
                    for j in range(i*i,n,i):
                        isprime[j] = False
            return sum(isprime)             
```

### 2，Surrounded Regions | DP

---
130. Surrounded Regions
https://leetcode.com/problems/surrounded-regions/