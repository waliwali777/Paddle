import unittest

import numpy as np
import paddle.v2.fluid.core as core
from paddle.v2.fluid.op import Operator


class TestBeamSearchDecodeOp(unittest.TestCase):
    def setUp(self):
        self.scope = core.Scope()
        self.cpu_place = core.CPUPlace()

    def append_lod_tensor(self, tensor_array, lod, data):
        lod_tensor = core.LoDTensor()
        lod_tensor.set_lod(lod)
        lod_tensor.set(data, self.cpu_place)
        tensor_array.append(lod_tensor)

    def test_get_set(self):
        ids = self.scope.var("ids").get_lod_tensor_array()
        self.append_lod_tensor(
            ids, [[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
            np.array(
                [1, 2, 3, 4, 5, 6], dtype="int64"))
        self.append_lod_tensor(
            ids, [[0, 3, 6], [0, 1, 1, 3, 5, 5, 6]],
            np.array(
                [0, 1, 2, 3, 4, 5], dtype="int64"))
        self.append_lod_tensor(
            ids, [[0, 3, 6], [0, 0, 1, 2, 3, 4, 5]],
            np.array(
                [0, 1, 2, 3, 4], dtype="int64"))

        scores = self.scope.var("scores").get_lod_tensor_array()
        self.append_lod_tensor(
            scores, [[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
            np.array(
                [1, 2, 3, 4, 5, 6], dtype="float64"))
        self.append_lod_tensor(
            scores, [[0, 3, 6], [0, 1, 1, 3, 5, 5, 6]],
            np.array(
                [0, 1, 2, 3, 4, 5], dtype="float64"))
        self.append_lod_tensor(
            scores, [[0, 3, 6], [0, 0, 1, 2, 3, 4, 5]],
            np.array(
                [0, 1, 2, 3, 4], dtype="float64"))

        sentence_ids = self.scope.var("sentence_ids").get_tensor()
        sentence_scores = self.scope.var("sentence_scores").get_tensor()

        beam_search_decode_op = Operator(
            "beam_search_decode",
            # inputs
            Ids="ids",
            Scores="scores",
            # outputs
            SentenceIds="sentence_ids",
            SentenceScores="sentence_scores")

        beam_search_decode_op.run(self.scope, self.cpu_place)

        expected_lod = [[0, 4, 8], [0, 1, 3, 6, 9, 10, 13, 16, 19]]
        self.assertEqual(sentence_ids.lod(), expected_lod)
        self.assertEqual(sentence_scores.lod(), expected_lod)

        expected_data = np.array(
            [2, 1, 0, 3, 1, 0, 3, 2, 1, 5, 4, 3, 2, 4, 4, 3, 6, 5, 4], "int64")
        self.assertTrue(np.array_equal(np.array(sentence_ids), expected_data))
        self.assertTrue(
            np.array_equal(np.array(sentence_scores), expected_data))


if __name__ == '__main__':
    unittest.main()
