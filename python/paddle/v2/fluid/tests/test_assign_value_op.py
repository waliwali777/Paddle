import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
import op_test
import numpy
import unittest
import paddle.v2.fluid.framework as framework


class TestAssignValueOp(op_test.OpTest):
    def setUp(self):
        self.op_type = "assign_value"
        x = numpy.random.random(size=(2, 5)).astype(numpy.float32)
        self.inputs = {}
        self.outputs = {'Out': x}
        self.attrs = {
            'shape': x.shape,
            'dtype': framework.convert_np_dtype_to_dtype_(x.dtype),
            'fp32_values': [float(v) for v in x.flat]
        }

    def test_forward(self):
        self.check_output()

    def test_assign(self):
        val = numpy.random.random(size=(2, 5)).astype(numpy.float32)
        x = layers.create_tensor(dtype="float32")
        layers.assign(input=val, output=x)
        exe = fluid.Executor(fluid.CPUPlace())
        fetched_x = exe.run(fluid.default_main_program(),
                            feed={},
                            fetch_list=[x])
        self.assertTrue(
            numpy.allclose(fetched_x, val),
            "fetch_x=%s val=%s" % (fetched_x, val))


if __name__ == '__main__':
    unittest.main()
