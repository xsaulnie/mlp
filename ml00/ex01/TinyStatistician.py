import sys
import numpy as np
import math
class TinyStatistician():
    @staticmethod

    def check_type(li):

        niteger = [int, float]
        if type(li) is list:
            for x in li:
                if (not type(x) in niteger):
                    return False
            return True
        if isinstance(li, np.ndarray):
            if (len(li.shape) != 1):
                return False
            if (li.dtype != np.int64 and li.dtype != np.float64):
                return False
            return True
        return False

    def mean(self, li):

        if not TinyStatistician.check_type(li):
            return None
        if (len(li) == 0):
            return None
        ret = 0
        for x in li:
            ret = ret + x
        return (float(ret / len(li)))

    def median(self, li):
        if not TinyStatistician.check_type(li):
            return None
        lng = len(li)
        if (lng < 1):
            return None
        sort = sorted(li)
        if lng % 2 == 1:
            return (float(sort[int((lng + 1)/ 2 - 1)]))
        else:
            return(float((sort[int(lng / 2 - 1)] + sort[int(lng / 2)]) / 2))

    def quartiles(self, li):
        if not TinyStatistician.check_type(li):
            return None
        lng = len(li)
        if (lng < 1): #or 4
            return None
        sort = sorted(li)
        
        r1 = math.ceil(lng / 4)
        r2 = math.ceil(lng / 4 * 3)
        return ([float(sort[int(r1 - 1)]), float(sort[int(r2 - 1)])])

        # if lng % 2 == 0:
        #     q1 = sort[:int(lng / 2)]
        #     q3 = sort[int(lng / 2):]
        # else:
        #     q1 = sort[:int(lng / 2)]
        #     q3 = sort[int(lng / 2) + 1:]
        # lq1 = len(q1)
        # lq3 = len(q3)

        # if (lq1 % 2 == 1):
        #     t1 = float(q1[int((lq1 + 1)/ 2 - 1)])
        # else:
        #     t1 = float((q1[int(lq1 / 2 - 1)] + q1[int(lq1 / 2)]) / 2)
        
        # if (lq3 % 2 == 1):
        #     t3 = float(q3[int((lq3 + 1)/ 2 - 1)])
        # else:
        #     t3 = float((q3[int(lq3 / 2 - 1)] + q3[int(lq3 / 2)]) / 2)
        # return ([t1, t3])
    def percentile(self, x, p):
        if not TinyStatistician.check_type(x):
            return None
        if not type(p) is int:
            return None
        lng = len(x)
        if lng == 0:
            return None
        if p > 100 or p < 0:
            return None
        if p == 0:
            return (x[0])
        if p == 100:
            return (x[lng - 1])
        sort = sorted(x)

        obs = (lng - 1) * (p / 100)
        ent = int(obs)
        weight = obs - ent

        return (float(round(sort[ent] + (weight * (sort[ent + 1] - sort[ent])), 2)))


    def var(self, li):
        if not TinyStatistician.check_type(li):
            return None
        lng = len(li)
        if (lng == 0):
            return None
        mean = self.mean(li)

        ret = 0
        for x in li:
            ret = ret + (x - mean)**2
        return (float(ret / (lng - 1)))
    def std(self, li):
        if not TinyStatistician.check_type(li):
            return None
        lng = len(li)
        if (lng == 0):
            return None
        mean = self.mean(li)

        ret = 0
        for x in li:
            ret = ret + (x - mean)**2
        return (float(math.sqrt(ret / (lng - 1))))

if __name__ == "__main__":
    tstat = TinyStatistician()

    # a = [1, 42, 300, 10, 59]
    # b = np.array([10, 20, 30, 40])

    # print("mean")
    # print(tstat.mean(a))
    # print(tstat.mean(b))

    # print("median")
    # print(tstat.median(a))
    # print(tstat.median(b))

    # print("quartiles")
    # print(tstat.quartiles(a))
    # print(tstat.quartiles(b))

    # print("percentile")
    # print(tstat.percentile(a, 50))
    # print(tstat.percentile(b, 25))
    # print(tstat.percentile(b, 75))
    # print(tstat.percentile(a,0))
    # print(tstat.percentile(b, 100))

    # print("variance")
    # print(tstat.var(a))
    # print(tstat.var(b))

    # print("standard deviation")
    # print(tstat.std(a))
    # print(tstat.std(b))

    print("test main")

    a = [1, 42, 300, 10, 59]
    print(a)
    print("mean")
    print(TinyStatistician().mean(a))
    print("median")
    print(TinyStatistician().median(a))
    print("quartiles")
    print(TinyStatistician().quartiles(a))
    print("percentiles")
    print(TinyStatistician().percentile(a, 10))
    #print("ref : ", np.percentile(np.array(a), 10))
    print(TinyStatistician().percentile(a, 15))
    #print("ref : ", np.percentile(np.array(a), 15))
    print(TinyStatistician().percentile(a, 20))
    #print("ref : ", np.percentile(np.array(a), 20))
    print(TinyStatistician().var(a))
    print(TinyStatistician().std(a))
    print("var")
    print(TinyStatistician().var(a))
    print("std")
    print(TinyStatistician().std(a))

