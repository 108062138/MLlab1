import math

def divAndCon(n,lv):
    if n==1:
        return [1]
    elif n==2:
        return [1,2]
    elif n==3:
        return [1,3,2]
    else:
        print('at lv: ',lv)
        res = []
        nodd = math.ceil(n/2)
        neve = math.floor(n/2)
        odds = divAndCon(nodd,lv+1)
        eves = divAndCon(neve,lv+1)
        for element in odds:
            print(2 * element-1)
            res.append(2 * element-1)
        for element in eves:
            print(2 * element)
            res.append(2 * element)
        return res
def sol(B,n):
    #lb =1 
    #rb =n
    #lastRb = n
    #lastLb  = 1
    #print(B[lb],B[rb])
    #while lb!=rb:
    #    if B[lb]>B[rb]:
    #        lastRb = rb
    #        rb = math.floor(rb/2)
    #    else:
    #        lb = rb 
    pivot = B[n]
    left = 1
    right = n
    while (left!=right):
        mid = math.floor((left+right)/2)
        if (B[mid] > pivot):
            left = mid + 1
        else:  
            right = mid
    return left

if __name__ == '__main__':
    #res = divAndCon(9,0)
    #print(res)
    A = [87,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6]
    #A = [87,4,5,6,7,8,9,10,1,2,3]
    #res = sol(A,15)
    print(A[-1])
    for i in range(1,5):
        print(i)