"""Rule behavior and complete-game legality, not strength benchmarks."""
import random
import unittest
from src.agent import HumanRulesAgent3, BasicAgent
from src.president import Game
from tests.test_president import ready_round, card

class HumanRulesAgent3Tests(unittest.TestCase):
    def choose(self, ranks, top=None):
        used={}; hand=[]
        for rank in ranks.split():
            i=used.get(rank,0);used[rank]=i+1;hand.append(card(rank,'CDHS'[i]))
        r=ready_round({'a':hand,'b':[card('7')],'c':[card('8')],'d':[card('9')]},top=top,last='d' if top else None)
        return HumanRulesAgent3().respond(r.message_for('a'))

    def test_singleton_lead(self):
        self.assertEqual(self.choose('3 3 4 8 K A')['cards'][0]['rank'],'4')

    def test_intact_response(self):
        a=self.choose('3 3 4 4 K K A 2',('9',2))
        self.assertEqual([c['rank'] for c in a['cards']],['K','K'])

    def test_finish_before_bomb_logic(self):
        self.assertEqual(len(self.choose('A A',('9',2))['cards']),2)

    def test_preserve_pair(self):
        self.assertEqual(self.choose('A A',('9',1))['type'],'pass')

    def test_bomb_before_last_group(self):
        self.assertEqual(self.choose('A A 2',('9',1))['type'],'bomb')

    def test_do_not_finish_with_bomb(self):
        self.assertEqual(self.choose('2',('9',1))['type'],'pass')

    def test_complete_games_and_trades(self):
        for seed in range(12):
            players=['a','b','c','d']; game=Game(players,rng=random.Random(seed))
            agents={p:HumanRulesAgent3() if i%2 else BasicAgent() for i,p in enumerate(players)}
            for _ in range(2):
                game.start_round();game.run(agents,max_actions=3000)

    def test_trade_requests_quad_and_skips_failed_rank(self):
        agent=HumanRulesAgent3()
        m={'player':'a','hand':[card('8',s).json() for s in 'CDH'] + [card('A').json()], 'events':[]}
        actions=[{'type':'trade_request','rank':r} for r in ('8','A','2')]
        self.assertEqual(agent._trade_rank(m,actions)['rank'],'8')
        m['events']=[{'type':'trade_requested','initiator':'a','rank':'8','success':False}]
        self.assertEqual(agent._trade_rank(m,actions)['rank'],'2')

    def test_trade_returns_singleton_before_breaking_pair(self):
        agent=HumanRulesAgent3()
        hand=[card('3',s).json() for s in 'CD']+[card('K').json()]
        actions=[{'type':'trade_return','card':c} for c in hand]
        self.assertEqual(agent._trade_return({'hand':hand},actions)['card']['rank'],'K')

    def test_trade_never_returns_two_while_single_exists(self):
        agent=HumanRulesAgent3()
        hand=[card('3').json(),card('2').json()]
        actions=[{'type':'trade_return','card':c} for c in hand]
        self.assertEqual(agent._trade_return({'hand':hand},actions)['card']['rank'],'3')

    def test_trade_returns_triple_card_without_singles(self):
        agent=HumanRulesAgent3()
        hand=[card('5',s).json() for s in 'CD']+[card('9',s).json() for s in 'CDH']
        actions=[{'type':'trade_return','card':c} for c in hand]
        self.assertEqual(agent._trade_return({'hand':hand},actions)['card']['rank'],'9')

    def test_trade_lone_ace_exception(self):
        agent=HumanRulesAgent3()
        hand=[card('2',s).json() for s in 'CD']+[card('A').json(),card('5').json()]
        actions=[{'type':'trade_request','rank':r} for r in ('5','A','2')]
        m={'player':'a','hand':hand,'events':[]}
        self.assertEqual(agent._trade_rank(m,actions)['rank'],'A')

    def test_trade_failed_two_pairs_highest_single(self):
        agent=HumanRulesAgent3()
        hand=[card('5').json(),card('Q').json(),card('9',"C").json(),card('9',"D").json()]
        actions=[{'type':'trade_request','rank':r} for r in ('5','9','Q','2')]
        m={'player':'a','hand':hand,
           'events':[{'type':'trade_requested','initiator':'a','rank':'2','success':False}]}
        self.assertEqual(agent._trade_rank(m,actions)['rank'],'Q')

    def test_lead_lowest_single_before_lower_pair(self):
        self.assertEqual(self.choose('3 3 7 K K')['cards'][0]['rank'],'7')

    def test_lead_splits_when_twos_cover_groups(self):
        a=self.choose('5 5 9 9 2 2')
        self.assertEqual((a['cards'][0]['rank'],len(a['cards'])),('5',1))

    def test_bomb_when_only_split_follow(self):
        self.assertEqual(self.choose('8 8 J J K K 2',('9',1))['type'],'bomb')

    def test_bomb_when_twos_plus_one_group(self):
        # G=3, B=2: G-B<=1 bombs even though a clean follow exists.
        self.assertEqual(self.choose('3 3 7 7 K K 2 2',('9',2))['type'],'bomb')

    def test_break_triple_against_single_without_two(self):
        a=self.choose('4 4 J J J',('9',1))
        self.assertEqual((a['type'],a['cards'][0]['rank']),('play','J'))

    def test_completion_declined_when_twos_lack_cover(self):
        agent=HumanRulesAgent3()
        hand=[card('9',s).json() for s in 'CD']+[card('K').json(),card('2').json()]
        actions=[{'type':'complete','cards':hand[:2]},{'type':'decline_completion'}]
        self.assertEqual(agent._completion({'hand':hand},actions)['type'],'decline_completion')
        hand.append(card('5').json())
        self.assertEqual(agent._completion({'hand':hand},actions)['type'],'complete')

if __name__=='__main__':
    unittest.main()
