
class A:
    a = 1
    def _init__(self):
        pass

b = A()
print(A.a)
print(hasattr(b, 'a'))
