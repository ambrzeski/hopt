import random


class Param(object):
    def randomize(self):
        raise NotImplementedError()


class Int(int, Param):
    def __new__(cls, a=None, b=None, value=0):
        obj = int.__new__(cls, value)
        obj.a = a
        obj.b = b
        return obj

    def randomize(self):
        if self.a is None or self.b is None:
            return self
        else:
            value = random.randint(self.a, self.b)
            new_param = Int(self.a, self.b, value)
            return new_param


class IntChoice(int, Param):
    def __new__(cls, values=None, value=0):
        obj = int.__new__(cls, value)
        obj.values = values
        return obj

    def randomize(self):
        if self.values is None:
            return self
        else:
            value = random.choice(self.values)
            new_param = IntChoice(self.values, value)
            return new_param


class FloatChoice(float, Param):
    def __new__(cls, values=None, value=0):
        obj = float.__new__(cls, value)
        obj.values = values
        return obj

    def randomize(self):
        if self.values is None:
            return self
        else:
            value = random.choice(self.values)
            new_param = FloatChoice(self.values, value)
            return new_param


class Float(float, Param):
    def __new__(cls, a=None, b=None, value=0):
        obj = float.__new__(cls, value)
        obj.a = a
        obj.b = b
        return obj

    def randomize(self):
        if None in [self.a, self.b]:
            return self
        else:
            value = random.uniform(self.a, self.b)
            new_param = Float(self.a, self.b, value)
            return new_param


class FloatExp(float, Param):
    def __new__(cls, base=0, pow_a=0, pow_b=0, k=1, value=0):
        obj = float.__new__(cls, value)
        obj.base = base
        obj.pow_a = pow_a
        obj.pow_b = pow_b
        obj.k = k
        return obj

    def randomize(self):
        if None in [self.k, self.base, self.pow_a, self.pow_b]:
            return self
        else:
            power = random.uniform(self.pow_a, self.pow_b)
            value = self.k * self.base ** power
            new_param = FloatExp(self.base, self.pow_a, self.pow_b, self.k, value)
            return new_param
