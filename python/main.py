# Cau a
ls = [3, 3, 5, 7, 3, 7]
b = int(input("Nhap so nguyen b: "))
# Cau b

if b > len(ls):
    print(f"{b} lớn hơn số lượng vị trí")
else:
    s = 0
    for i in range(b + 1):
        s += ls[i]
    print(s)
    