def assert_almost_equal(a, b, eps=1e-7):
    for x, y in zip(a, b):
        assert abs(x - y) <= eps, '{} != {}'.format(x, y)


