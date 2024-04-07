from inspect import isfunction


"""
===========================1.==============================================

"""
def exists(x):
    """
    判断输入的x 是否为 None
    :param x: 输入的 需要判断的值
    :return:  赶回True 或者 False
    """
    return x is not None


def default(val, d):
    """
    1. 该函数判断val 是否为None   若不为None 则将其返回
    2. 判断d是否为函数,若为函数 返回函数不为函数 说明是变量,把他当作变量返回
    :param val:
    :param d:
    :return:
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d