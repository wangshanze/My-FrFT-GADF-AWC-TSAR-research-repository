import numpy
import scipy
import scipy.signal


def frft(f, a):
    """
    Calculate the fast fractional fourier transform.

    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power

    Returns
    -------
    data : numpy array
        The transformed signal.
    """
    ret = numpy.zeros_like(f, dtype=numpy.complex128)
    f = f.copy().astype(numpy.complex128)
    N = len(f)
    shft = numpy.fmod(numpy.arange(N) + numpy.fix(N / 2), N).astype(int)
    sN = numpy.sqrt(N)
    a = numpy.remainder(a, 4.0)

    # Special cases
    if a == 0.0:
        return f
    if a == 2.0:
        return numpy.flipud(f)
    if a == 1.0:
        ret[shft] = numpy.fft.fft(f[shft]) / sN
        return ret
    if a == 3.0:
        ret[shft] = numpy.fft.ifft(f[shft]) * sN
        return ret

    # reduce to interval 0.5 < a < 1.5
    if a > 2.0:
        a = a - 2.0
        f = numpy.flipud(f)
    if a > 1.5:
        a = a - 1
        f[shft] = numpy.fft.fft(f[shft]) / sN
    if a < 0.5:
        a = a + 1
        f[shft] = numpy.fft.ifft(f[shft]) * sN

    # the general case for 0.5 < a < 1.5
    alpha = a * numpy.pi / 2
    tana2 = numpy.tan(alpha / 2)
    sina = numpy.sin(alpha)
    f = numpy.hstack((numpy.zeros(N - 1), sincinterp(f), numpy.zeros(N - 1))).T

    # chirp premultiplication
    chrp = numpy.exp(-1j * numpy.pi / N * tana2 / 4 *
                     numpy.arange(-2 * N + 2, 2 * N - 1).T ** 2)
    f = chrp * f

    # chirp convolution
    c = numpy.pi / N / sina / 4
    ret = scipy.signal.fftconvolve(
        numpy.exp(1j * c * numpy.arange(-(4 * N - 4), 4 * N - 3).T ** 2),
        f
    )
    ret = ret[4 * N - 4:8 * N - 7] * numpy.sqrt(c / numpy.pi)

    # chirp post multiplication
    ret = chrp * ret

    # normalizing constant
    ret = numpy.exp(-1j * (1 - a) * numpy.pi / 4) * ret[N - 1:-N + 1:2]

    return ret


def ifrft(f, a):
    """
    Calculate the inverse fast fractional fourier transform.

    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power

    Returns
    -------
    data : numpy array
        The transformed signal.

    """
    return frft(f, -a)


def sincinterp(x):
    N = len(x)
    y = numpy.zeros(2 * N - 1, dtype=x.dtype)
    y[:2 * N:2] = x
    xint = scipy.signal.fftconvolve(
        y[:2 * N],
        numpy.sinc(numpy.arange(-(2 * N - 3), (2 * N - 2)).T / 2),
    )
    return xint[2 * N - 3: -2 * N + 3]


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    a = numpy.arange(0, 4.25, 0.25)  # 分数阶傅里叶变换阶数

    # 生成一个窗函数
    fx = numpy.zeros(500)
    fx[150:251] = 1

    for ai in a:
        plt.figure()
        F = frft(fx, ai)
        plt.plot(numpy.abs(F))
        plt.title(f"a={ai}")
        plt.grid(True)
        plt.ylim(0, 5)
        plt.show()