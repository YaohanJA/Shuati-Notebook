title1: "Python OOP"
date: 2020-04-11



## Python OOP
### Defining Classes
[Problem Solving with Algorithms and Data Structures using Python](https://runestone.academy/runestone/books/published/pythonds/Introduction/ObjectOrientedProgramminginPythonDefiningClasses.html#object-oriented-programming-in-python-defining-classes)

#### 1,`__init__(self)`

```python
class Fraction:
		
    def __init__(self,top,bottom):
        self.num = top
        self.den = bottom
```

- `self` is a special parameter that will always be used as a reference back to the object itself. It must always be the first formal parameter; however, it will never be given an actual parameter value upon invocation. 
	
   ```python
   myfraction = Fraction(3,5)
   ```
#### 2,`__str__` `print`
```python
>>> myf = Fraction(3,5)
>>> print(myf)
<__main__.Fraction instance at 0x409b1acc>
```
- The `print` function requires that the object convert itself into a string so that the string can be written to the output. 

```python
def show(self):
    print(self.num, "/", self.den)

myf = Fraction(2,3)
myf.show()
```

- The `__str__` is the method to convert an object into a string.

  The default implementation for this method is to return the instance address string as we have already seen. `<__main__.Fraction instance at 0x409b1acc>`

```python
def __str__(self):
    return str(self.num)+"/"+str(self.den)

print(myf) #3/5
```

#### 3,`__add__` 
```python
def __add__(self, otherfraction):
	newnum = self.num*otherfraction.den + self.den*otherfraction.num
  newden = self.den * otherfraction.den
  
  return Fraction(newnum,newden)

>>> f1=Fraction(1,4)
>>> f2=Fraction(1,2)
>>> f3=f1+f2
>>> print(f3)
6/8
```

- 6/8 是对的，但是不是最大公约数3/4。

- The best-known algorithm for finding a greatest common divisor [GCD] is **Euclid’s Algorithm** 

##### 欧几里德算法

- 求最大公约数
  ```python
  def gcd(a, b):
      while b != 0:
          a , b = b, a % b
      return a 
  ```
- new `__add__` 
 ```python
  def __add__(self,otherfraction):
    newnum = self.num*otherfraction.den + self.den*otherfraction.num
    newden = self.den * otherfraction.den
    common = gcd(newnum,newden)
    return Fraction(newnum//common,newden//common)
  
 >>> f1=Fraction(1,4)
>>> f2=Fraction(1,2)
>>> f3=f1+f2
>>> print(f3)
3/4
 ```

#### 3,`__eq__`  
- The `__eq__` method compares two objects and returns True if their values are the same, False otherwise.

## **shallow equality** !!!!





