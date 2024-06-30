import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from ..env.pnpsc_net import PnpscNet

# Flag from the PNPSC specification
RESET = True
# Do players incur the fire cost of transitions
USE_FIRE_COST = False
# Do control rate modifications persist after the step
RESET_CONTROL_RATE = False
# Large time in the future for disabled rates
LARGE_TIME = 100

class Simulator():
    """
    Local implementation of the PNPSC net simulator using NetworkX
    """
    def __init__(self, net):
        """
        Create a PNPSC net simulator
        :param net: PNPSC net object
        """
        self.net = net
        self.g = nx.DiGraph()

        # used for rendering
        self.places = [p['name'] for p in self.net.json['places']]
        self.transitions = [t['name'] for t in self.net.json['transitions']]
        self.inhibitors = [(t['inhibitor'], t['name']) for t in self.net.json['transitions'] if t['inhibitor'] != '']

        self.control_rates = {t['name']: [] for t in self.net.json['transitions']}

        self.fire_cost = {t['name']: (t['player_control'], t['fire_cost']) for t in sorted(self.net.json['transitions'],
                                                                                  key=lambda x: x['name'])}

        for p in self.net.json['places']:
            self.g.add_node(p['name'], type='place', control=p['player_observable'],
                            explanation=p['description'] if 'description' in p else '')

        for t in self.net.json['transitions']:
            self.g.add_node(t['name'], type='transition', control=t['player_control'],
                            explanation=t['description'] if 'description' in t else '')
            for ti in t['input'].split(','):
                self.g.add_edge(ti, t['name'], weight=1)
            for to in t['output'].split(','):
                self.g.add_edge(t['name'], to, weight=1)
            if t['inhibitor'] != '':
                for ia in t['inhibitor'].split(','):
                    self.g.add_edge(ia, t['name'], weight=-1)
            for cr in t['control_rate'].split(','):
                if cr != '':
                    p, r = cr.split('=')
                    self.control_rates[t['name']].append((p, int(r)))

        self.t = 0
        self.fired = None
        self.ft = np.full(len(self.net.rates.keys()), np.inf)
        self.updated = []
        self.reset()

    def reset(self):
        """
        Reset the simulator to the original marking and rates
        """
        self.net.done = False
        self.net.costs = {p: 0 for p in self.net.players}
        self.net.places = {p['name']: p['marking'] for p in sorted(self.net.json['places'], key=lambda x: x['name'])}
        self.net.rates = {t['name']: t['rate'] for t in sorted(self.net.json['transitions'], key=lambda x: x['name'])}
        self.t = 0
        self.fired = None
        self.ft = np.full(len(self.net.rates.keys()), np.inf)
        self.updated = []

    def update_rates(self, rates):
        """
        Update the transition rates
        :param rates: Dictionary of rates to update
        """
        self.updated = []
        for k, v in rates.items():
            self.updated.append(k)
            self.net.rates[k] = v

    def _check_enabled(self):
        """
        Returns the enabled transition for the current marking
        :return: A list of enabled transitions
        """
        enabled = [True for t in self.net.get_all_rates().keys()]
        for i, t in enumerate(self.net.get_all_rates().keys()):
            for e in self.g.in_edges(t, data=True):
                if e[2]['weight'] == -1:
                    # inhibitor
                    if self.net.places[e[0]] > 0:
                        enabled[i] = False
                elif self.net.places[e[0]] <= 0:
                    enabled[i] = False
        return enabled

    def step(self):
        """
        Step the PNPSC net simulator
        """
        enabled = self._check_enabled()

        rates = self.net.get_all_rates()
        if any(enabled):
            for i, t in enumerate(rates):
                rate = rates[t]
                # set control rates
                for p, rv in self.control_rates[t]:
                    if self.net.places[p] > 0:
                        rate += rv

                if enabled[i]:
                    if RESET or self.ft[i] == np.inf:
                        if rate == 0:
                            # to mimic the cloud sim, if the rate is 0 pick a time far into the future
                            self.ft[i] = LARGE_TIME + self.t
                        else:
                            # update firing time
                            self.ft[i] = np.random.exponential(1 / rate) + self.t
                else:
                    # not enabled
                    self.ft[i] = np.inf
                if RESET_CONTROL_RATE:
                    rates[t] = rate

            # to mimic the cloud sim, pick the first transition if all are the same
            j = np.argmin(self.ft)
            # selected transition to fire
            self.t = self.ft[j]
            self.fired = j
            self.ft[j] = np.inf
            fired_named = list(self.net.rates.keys())[j]
            for e in self.g.in_edges(fired_named, data=True):
                if e[2]['weight'] >= 1:
                    self.net.places[e[0]] -= 1
            for e in self.g.out_edges(fired_named):
                self.net.places[e[1]] += 1

            player, cost = self.fire_cost[fired_named]

            if player is not None and player != 'None' and player != '':
                self.net.costs[player] += cost if cost is not None and USE_FIRE_COST else 0
        else:
            self.net.done = True

    def render(self):
        """
        Render the PNPSC net state using NetworkX with Graphviz format
        """
        # plt.figure(figsize=(12,8))
        plt.clf()
        pos = nx.nx_agraph.graphviz_layout(self.g)

        e = self._check_enabled()
        enabled = [t for i, t in enumerate(self.net.rates.keys()) if e[i]]

        # draw places and edges
        nx.draw(self.g, nodelist=self.places, pos=pos, node_shape="o", edge_color='black', node_size=1200,
                labels={n: n + '\n' + str((self.net.places[n]) if n in self.net.places else self.net.rates[n]) for n in
                        self.g.nodes()})
        # draw all transitions
        nx.draw_networkx_nodes(self.g, nodelist=self.transitions, pos=pos, node_shape='s', node_size=1000,
                               node_color='grey')

        # draw enabled transitions
        nx.draw_networkx_nodes(self.g, nodelist=enabled, pos=pos, node_shape='s', node_size=800,
                               node_color='lightblue')
        # draw updated transitions
        nx.draw_networkx_nodes(self.g, nodelist=self.updated, pos=pos, node_shape='s', node_size=800,
                               node_color='purple')
        # draw fired transitions
        # TODO Add a None check on fired
        nx.draw_networkx_nodes(self.g, nodelist=[list(self.net.rates.keys())[self.fired]], pos=pos, node_shape='s',
                               node_size=800, node_color='red')

        nx.draw_networkx_edges(self.g, edgelist=self.inhibitors, pos=pos, edge_color='red', width=1.5, arrows=True)

        plt.axis('off')
        plt.pause(2)
        # write_dot(self.g, 'file.dot')

    def output_net(self, path, show_desc):
        g = nx.DiGraph(self.g)
        for id, attrib in g.nodes.items():
            g.nodes[id]['name'] = g.nodes[id]['explanation'] if show_desc else id
            for p in self.net.players:
                if id in self.net.get_goal_places(p):
                    g.nodes[id]['control'] = p + '_goal'

        for id, attrib in g.edges.items():
            g.edges[id]['weight'] = np.int32(attrib['weight'])

        nx.write_graphml(g, path, named_key_ids=True)
