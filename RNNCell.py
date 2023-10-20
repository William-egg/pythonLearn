import sympy
y = sympy.Symbol('y')
A = sympy.Matrix([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1]]
)
B = sympy.Matrix([
    [0],
    [1],
    [1],
    [1]]
)
E = sympy.eye(4)
EY = sympy.Matrix([
    [y, 0, 0, 0],
    [0, y, 0, 0],
    [0, 0, y, 0],
    [0, 0, 0, y]]
)
result = E-EY*A
print(result.inv()*B)