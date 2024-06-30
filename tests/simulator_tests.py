import json
import unittest

from src.pnpsc_env.env.pnpsc_net import PnpscNet
from src.pnpsc_env.simulator.simulator import Simulator


class TestSimulator(unittest.TestCase):

    def test_simulator_fire(self):
        with open('../nets/example_net.json') as f:
            data = json.load(f)
            net = PnpscNet(data)
            s = Simulator(net)

            self.assertEqual({'aP1': 10, 'aP2': 0, 'aP3': 0, 'aP4': 0, 'aP5': 0}, s.net.get_all_places())

            s.update_rates({'aT1': 0, 'aT2': 10, 'aT3': 0, 'aT4': 0})
            s.step()

            self.assertEqual({'aP1': 9, 'aP2': 1, 'aP3': 1, 'aP4': 0, 'aP5': 0}, s.net.get_all_places())

    def test_rate_update(self):
        with open('../nets/example_net.json') as f:
            data = json.load(f)
            net = PnpscNet(data)
            s = Simulator(net)

            self.assertEqual({'aP1': 10, 'aP2': 0, 'aP3': 0, 'aP4': 0, 'aP5': 0}, s.net.get_all_places())
            self.assertEqual({'aT1': 10, 'aT2': 5, 'aT3': 10, 'aT4': 2}, s.net.get_all_rates())

    def test_control_rate(self):
        with open('../nets/example_net.json') as f:
            data = json.load(f)
            net = PnpscNet(data)
            s = Simulator(net)

            s.net.places = {'aP1': 9, 'aP2': 0, 'aP3': 1, 'aP4': 0, 'aP5': 0}
            s.update_rates({'aT1': 0, 'aT2': 0, 'aT3': 0, 'aT4': 0})
            s.step()

            self.assertEqual({'aP1': 9, 'aP2': 0, 'aP3': 0, 'aP4': 1, 'aP5': 0}, s.net.get_all_places())

    def test_inhibitor(self):
        with open('../nets/example_net.json') as f:
            data = json.load(f)
            net = PnpscNet(data)
            s = Simulator(net)

            s.net.places = {'aP1': 8, 'aP2': 1, 'aP3': 0, 'aP4': 1, 'aP5': 1}
            s.update_rates({'aT1': 10, 'aT2': 10, 'aT3': 0, 'aT4': 0})
            s.step()

            self.assertEqual({'aP1': 7, 'aP2': 1, 'aP3': 1, 'aP4': 1, 'aP5': 1}, s.net.get_all_places())


    def test_simulator_full(self):
        """
        Test the basic functionality of the local environment
        """
        with open('../nets/example_net.json') as f:
            data = json.load(f)
            net = PnpscNet(data)
            s = Simulator(net)

            # -- STEP 0
            self.assertEqual({'aP1': 10, 'aP2': 0, 'aP3': 0, 'aP4': 0, 'aP5': 0}, s.net.get_all_places())
            self.assertEqual({'aT1': 10, 'aT2': 5, 'aT3': 10, 'aT4': 2}, s.net.get_all_rates())

            # -- STEP 1
            s.update_rates({'aT1': 10, 'aT2': 0, 'aT3': 0, 'aT4': 0})
            self.assertEqual({'aT1': 10, 'aT2': 0, 'aT3': 0, 'aT4': 0}, s.net.get_all_rates())
            enabled = s._check_enabled()
            self.assertEqual([True, True, False, False], enabled)
            s.step()

            # -- STEP 2
            self.assertEqual({'aP1': 9, 'aP2': 0, 'aP3': 1, 'aP4': 0, 'aP5': 0}, s.net.get_all_places())
            # tests control rate for aT3
            s.update_rates({'aT1': 0, 'aT2': 0, 'aT3': 0, 'aT4': 0})
            self.assertEqual({'aT1': 0, 'aT2': 0, 'aT3': 0, 'aT4': 0}, s.net.get_all_rates())
            enabled = s._check_enabled()
            self.assertEqual([True, True, True, True], enabled)
            s.step()

            # -- STEP 3
            self.assertEqual({'aP1': 9, 'aP2': 0, 'aP3': 0, 'aP4': 1, 'aP5': 0}, s.net.get_all_places())
            s.update_rates({'aT1': 0, 'aT2': 10, 'aT3': 0, 'aT4': 0})
            self.assertEqual({'aT1': 0, 'aT2': 10, 'aT3': 0, 'aT4': 0}, s.net.get_all_rates())
            enabled = s._check_enabled()
            self.assertEqual([True, True, False, False], enabled)
            s.step()

            # -- STEP 4
            self.assertEqual({'aP1': 8, 'aP2': 1, 'aP3': 1, 'aP4': 1, 'aP5': 0}, s.net.get_all_places())
            s.update_rates({'aT1': 0, 'aT2': 0, 'aT3': 0, 'aT4': 10})
            self.assertEqual({'aT1': 0, 'aT2': 0, 'aT3': 0, 'aT4': 10}, s.net.get_all_rates())
            enabled = s._check_enabled()
            self.assertEqual([True, True, True, True], enabled)
            # Disable the control rate to force aT4 to fire
            s.control_rates['aT3'] = []
            s.step()

            # -- STEP 5
            self.assertEqual({'aP1': 8, 'aP2': 1, 'aP3': 0, 'aP4': 1, 'aP5': 1}, s.net.get_all_places())
            s.update_rates({'aT1': 10, 'aT2': 10, 'aT3': 0, 'aT4': 0})
            self.assertEqual({'aT1': 10, 'aT2': 10, 'aT3': 0, 'aT4': 0}, s.net.get_all_rates())
            enabled = s._check_enabled()
            # aT2 is disabled due to inhibitor arc from aP5
            self.assertEqual([True, False, False, False], enabled)
            s.step()

            # -- STEP 6
            self.assertEqual({'aP1': 7, 'aP2': 1, 'aP3': 1, 'aP4': 1, 'aP5': 1}, s.net.get_all_places())


if __name__ == '__main__':
    unittest.main()
