import unittest
from src.model import main
from time import time


class TestModelTraining(unittest.TestCase):
    'Check can overfit small batch etc. without issues'

    def test_fast_dev_run(self):
        t = time()
        try:
            main(fast_dev_run = True)
        except:
            self.fail("Obvious Training Problem!")
        time_taken = time() - t
        self.assertLess(time_taken, 10)


if __name__ == '__main__':
    unittest.main()
